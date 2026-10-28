"""C7: are our images among real DROID frames, as the policy's own encoder sees them? (REAL2SIM_CHECKLIST_V1 C7, old 6.3;
implres31, 10-28.)

Two stages, two venvs:
  --dump   (.venvs/real2sim, CPU)  for the scene's pre-touch frames (frame 0 + episode_info 'before_touch'): our
           composites (exterior + WristRenderV2, the policy's 180x320) and the real frames of the same moments; plus a
           reference set of real frames from every local DROID episode (--ref_episodes, --ref_frames each). -> <out>/c7_images.npz
  --embed  (.venvs/groot, GPU by default: prim's go + disk line)  GR00T-N1.7-DROID loaded as policy_rollout --serve does;
           for each (exterior, wrist) pair, the backbone's last-layer features (what the action head reads) are averaged
           over each camera's image tokens (tokens split by image_grid_thw; Qwen spatial merge 2x2). Distance = 1 - cos.
           -> <out>/encoder_distance.json
Reported per camera: ours-vs-real at the same moment; the real-real scales (consecutive pre-touch frames of one
episode = motion; frames of different episodes = scene change); and an out-of-distribution percentile: our image's
nearest-neighbour distance to the reference set vs the reference frames' own leave-one-out nearest-neighbour distances.
Limits: tokens are contextualised by the instruction (one fixed --instruction for every image) and, for the wrist, by
the exterior image before it (causal attention): pairs are always (ours, ours) or (real, real). Six local episodes
are a small stand-in for "real DROID frames". No bar (Q1). Exploratory, UNVERIFIED.

    MUJOCO_GL=osmesa /home/imerit/Documents/Code/VLA/.venvs/real2sim/bin/python experiments/real2sim_checks/encoder_distance.py --dump \
        --scene /home/imerit/Documents/Code/VLA/runs/real2sim/ep_orange_block_stereo --episode /home/imerit/Documents/Code/VLA/data/real2sim/droid/ep_orange_block
    /home/imerit/Documents/Code/VLA/.venvs/groot/bin/python experiments/real2sim_checks/encoder_distance.py --embed \
        --out /home/imerit/Documents/Code/VLA/runs/real2sim_checks/ep_orange_block_stereo [--device cpu]
"""
import argparse, glob, json, os, sys, time
import numpy as np

MAIN = "/home/imerit/Documents/Code/VLA"
R2S = os.path.join(MAIN, "experiments/real2sim")


def dump(a, out):
    import common  # noqa: F401  (real2sim venv only)
    from common import SceneFrames, composites, to_uint8, POLICY_HW
    sf = SceneFrames(a.scene, a.episode)
    ec, wc, note = composites(sf, POLICY_HW)
    if ec is None:
        raise SystemExit("not run: " + note)
    ours, real, frames = [], [], []
    for f in sf.frames():
        rf = sf.pose(f)
        ours.append([to_uint8(ec(sf.d)), to_uint8(wc(sf.d))])
        real.append([sf.real("ext", rf, POLICY_HW), sf.real("wrist", rf, POLICY_HW)])
        frames.append(f)
    import cv2
    from episode_info import episode_info
    ref, ref_ep, ref_fr = [], [], []
    for ep in sorted(glob.glob(os.path.join(a.ref_root, "ep_*"))):
        try:
            info = episode_info(ep)
        except Exception as e:  # noqa: BLE001  (an episode without trajectory.h5 is skipped, and listed)
            print("skip reference", ep, e)
            continue
        sers = (info["ext1_cam_serial"], info["wrist_cam_serial"])
        if not all(s and os.path.exists(os.path.join(ep, f"{s}.mp4")) for s in sers):
            print("skip reference", ep, "camera videos missing", sers)
            continue
        for f in sorted({int(x) for x in np.linspace(0, info["n_frames"] - 1, a.ref_frames)}):
            pair = []
            for s in sers:
                cap = cv2.VideoCapture(os.path.join(ep, f"{s}.mp4"))
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, bgr = cap.read()
                cap.release()
                if not ok:
                    break
                pair.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (POLICY_HW[1], POLICY_HW[0]), interpolation=cv2.INTER_AREA))
            if len(pair) == 2:
                ref.append(pair)
                ref_ep.append(os.path.basename(ep))
                ref_fr.append(f)
    # the exterior camera the scene was built from may be ext1 or ext2: the reference uses each episode's ext1
    np.savez_compressed(os.path.join(out, "c7_images.npz"), ours=np.array(ours), real=np.array(real), frames=np.array(frames),
                        ref=np.array(ref), ref_episode=np.array(ref_ep), ref_frame=np.array(ref_fr),
                        scene_episode=os.path.basename(os.path.normpath(a.episode)), composite=note, ext_serial=sf.cams["ext"])
    print(f"dumped {len(frames)} scene frames, {len(ref)} reference pairs from {len(set(ref_ep))} episodes -> {out}/c7_images.npz")


class Encoder:
    """GR00T-N1.7-DROID up to its backbone, built exactly as policy_rollout.serve (bf16)."""

    def __init__(self, ckpt, device):
        sys.path.insert(0, R2S)
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors_from_pretrained
        from policy_rollout import IMG, H, W, TAG
        self.torch, self.IMG, self.device = torch, IMG, device
        feats = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)) for k in IMG}
        feats["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(17,))
        cfg = GrootConfig(base_model_path=ckpt, embodiment_tag=TAG, input_features=feats,
                          output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(17,))},
                          model_params_fp32=False, device="cpu")
        with torch.no_grad():
            policy = GrootPolicy(cfg)
        self.policy = policy.to(torch.bfloat16).to(device).eval()
        cfg.device = device
        self.pre, _ = make_groot_pre_post_processors_from_pretrained(cfg, ckpt, preprocessor_overrides={"device_processor": {"device": device}})

    def __call__(self, ext, wrist, instruction):
        """Per-camera pooled image-token features (float32, L2-normalised) and how the tokens were split."""
        torch = self.torch
        batch = {k: torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
                 for k, img in zip(self.IMG, (ext, wrist))}
        batch["observation.state"] = torch.zeros(1, 17, device=self.device)  # the backbone does not read the state
        batch["task"] = [instruction]
        model = self.policy._groot_model
        with torch.inference_mode(), torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16):
            gi = self.policy._filter_groot_inputs(self.pre(batch), include_action=False)
            bb_in, _ = model.prepare_input(gi)
            o = model.backbone(bb_in)
        f = o["backbone_features"][0].float()
        idx = torch.nonzero(o["image_mask"][0]).flatten()
        thw = bb_in.get("image_grid_thw")
        counts = None
        if thw is not None:
            counts = [int(x) // 4 for x in thw.reshape(-1, 3).prod(-1).tolist()]  # Qwen-VL spatial merge 2x2
        if counts and len(counts) == 2 and sum(counts) == len(idx):
            parts, how = [idx[:counts[0]], idx[counts[0]:]], f"split by image_grid_thw {counts}"
        else:
            parts, how = [idx, idx], f"NOT split (image tokens {len(idx)}, grid counts {counts}): both cameras pooled together"
        emb = []
        for p in parts:
            v = f[p].mean(0)
            emb.append((v / v.norm()).cpu().numpy())
        return emb, how


def embed(a, out):
    z = np.load(os.path.join(out, "c7_images.npz"))
    enc = Encoder(a.ckpt, a.device)
    t0 = time.time()

    def run(pairs):
        E, how = [], None
        for ext, wr in pairs:
            e, how = enc(ext, wr, a.instruction)
            E.append(e)
        return np.array(E), how  # (n, 2 cameras, D)

    Eo, how = run(z["ours"])
    Er, _ = run(z["real"])
    Eref, _ = run(z["ref"])
    dist = lambda x, y: 1.0 - np.clip((x * y).sum(-1), -1, 1)
    ref_ep, scene_ep = z["ref_episode"], str(z["scene_episode"])
    res = dict(scene=a.scene or out, instruction=a.instruction, token_split=how, device=a.device, seconds=None,
               status="exploratory, UNVERIFIED", cameras={})
    for ci, cam in enumerate(("ext", "wrist")):
        same = dist(Eo[:, ci], Er[:, ci])
        motion = dist(Er[1:, ci], Er[:-1, ci]) if len(Er) > 1 else np.array([])
        D = dist(Eref[:, None, ci], Eref[None, :, ci])
        cross = D[ref_ep[:, None] != ref_ep[None, :]]
        np.fill_diagonal(D, np.inf)
        loo = D.min(1)  # each reference frame's nearest other reference frame
        nn_ours = dist(Eo[:, None, ci], Eref[None, :, ci]).min(1)
        res["cameras"][cam] = dict(
            ours_vs_real_same_moment=[round(float(x), 5) for x in same],
            ours_vs_real_median=round(float(np.median(same)), 5),
            real_motion_median=None if not len(motion) else round(float(np.median(motion)), 5),
            real_cross_episode_median=None if not len(cross) else round(float(np.median(cross)), 5),
            ours_nn_to_reference=[round(float(x), 5) for x in nn_ours],
            reference_nn_loo_median=round(float(np.median(loo)), 5),
            ours_nn_percentile_among_reference=[round(float((loo <= x).mean() * 100), 1) for x in nn_ours],
            note=f"reference: {len(Eref)} real pairs from {sorted(set(ref_ep.tolist()))}; the scene's own episode {scene_ep} is in it")
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(out, "encoder_distance.json"), "w"), indent=1)
    for cam, v in res["cameras"].items():
        print(cam, "ours-real", v["ours_vs_real_median"], "| real motion", v["real_motion_median"], "| real cross-episode",
              v["real_cross_episode_median"], "| ours NN percentile", v["ours_nn_percentile_among_reference"])
    print(f"token split: {how}; written {out}/encoder_distance.json ({res['seconds']} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--out", default=None, help="default: <main checkout>/runs/real2sim_checks/<scene name>")
    ap.add_argument("--ref_root", default=os.path.join(MAIN, "data/real2sim/droid"))
    ap.add_argument("--ref_frames", type=int, default=8, help="reference frames per local episode")
    ap.add_argument("--instruction", default="pick up the object")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-DROID/snapshots/"
                                                         "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5"))  # policy_rollout.DEFAULT_CKPT
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--real2sim", default=None, help="res31's experiments/real2sim, imported read-only (read by common.py)")
    a = ap.parse_args()
    out = a.out or os.path.join(MAIN, "runs/real2sim_checks", os.path.basename(os.path.normpath(a.scene)))
    os.makedirs(out, exist_ok=True)
    if a.dump:
        dump(a, out)
    if a.embed:
        embed(a, out)


if __name__ == "__main__":
    main()
