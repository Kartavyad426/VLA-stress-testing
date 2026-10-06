"""S2 rescue minting (docs/DATA_MINER_SPEC.html §3, §4).

Per (scene, noise seed) the NOMINAL episode's encoded features are recorded
for the first `drive_until` forwards (attach_recorder, as R-044/R-047). Each
perturbed episode then runs with the splice driving N -- the head's output on
those nominal features, under the same fixed noise -- for forwards
< drive_until, after which the unmodified policy acts. The episode is recorded
per step (the env writes both cameras as PNG), and becomes one EpisodeRecord:

  * inputs are the perturbed episode's own images and proprioception, always;
  * frames before the forward `drive_until` starts are tagged driven=True
    (16 frames at drive_until=1, n_action_steps=16);
  * failures are written too, with exportable=False.

The splice and recorder are GR00T N1.7 structure (capture/splice.py). A
policy the splice cannot wrap cannot be a rescue source; everything else here
(records, provenance, held-out refusal) is policy-agnostic.
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np

from ...schema import PerturbationSpec, TraceStore
from ..contract import EpisodeRecord, EpisodeStore, FrameRecord, HeldOut, Provenance

# env-native proprioception carried into records; never a `_gt_` key
PROPRIO_KEYS = ("eef_pos", "eef_quat", "gripper_qpos", "joint_pos")
LIVE_DIMS = 7


def noise_key_for(noise_seed: int) -> Callable[[int], int]:
    """The harness's fixed-noise convention (r039_run / r047_run): 1000*seed + forward."""
    return lambda i, s=int(noise_seed): 1000 * s + i


def frames_from_rollout(r, store: EpisodeStore, driven_until_step: int) -> list[FrameRecord]:
    """Rollout steps -> FrameRecords. Image refs become paths relative to the store root,
    keyed by the env's camera name ("image", "image2"); `_gt_` keys are dropped (I2)."""
    out = []
    for st in r.steps:
        imgs = {}
        for k, p in (st.image_refs or {}).items():
            cam = k[3:] if k.startswith("cam") else k
            imgs[cam] = os.path.relpath(p, store.root) if os.path.isabs(p) or os.path.exists(p) else p
        out.append(FrameRecord(
            t=int(st.t), images=imgs,
            proprio={k: [float(x) for x in st.obs_state[k]] for k in PROPRIO_KEYS if k in st.obs_state},
            action=None if st.action is None else [float(x) for x in st.action],
            driven=(st.action is not None and int(st.t) < driven_until_step)))
    return out


def closest_to_target(r, env) -> float | None:
    """Analysis only: min eef distance to the target bowl (R-038/R-047's discriminator)."""
    try:
        targets = [t for t in env._env.unwrapped._env.obj_of_interest if "bowl" in t]
    except Exception:
        targets = []
    best = None
    for st in r.steps:
        dd = st.obs_state.get("_gt_eef_to_object") or {}
        pool = {k: v for k, v in dd.items() if any(k.startswith(t) or t.startswith(k) for t in targets)} or dd
        if pool:
            m = min(pool.values())
            best = m if best is None else min(best, m)
    return None if best is None else float(best)


def failure_family(r) -> str | None:
    try:
        from ...mining.classify import classify
        return classify(r).get("family")
    except Exception as e:                       # the miner abstains on some LIBERO traces
        return f"unclassified:{type(e).__name__}"


class RescueMinter:
    """Mint rescue episodes with one loaded policy.

    `policy` is a LeRobotPolicy (the harness wrapper); `make_env(scene_task_id)`
    builds a fresh LiberoEnv; `store` receives the records. `heldout` refuses
    evaluation starts (G4) before any compute is spent.
    """

    source = "rescue"

    def __init__(self, policy, make_env: Callable[[int], object], store: EpisodeStore,
                 heldout: HeldOut | None = None, drive: str = "N", drive_until: int = 1,
                 feature_cache: str | None = None, trace_store: bool = True):
        self.pol = policy
        self.make_env = make_env
        self.store = store
        self.heldout = heldout or HeldOut()
        self.drive = drive
        self.drive_until = int(drive_until)
        self.feature_cache = feature_cache or os.path.join(store.root, "nominal_features")
        os.makedirs(self.feature_cache, exist_ok=True)
        self._nominal: dict[tuple, list[dict]] = {}
        self.traces = TraceStore(os.path.dirname(os.path.abspath(store.root)),
                                 os.path.basename(os.path.abspath(store.root))) if trace_store else None
        os.makedirs(os.path.join(store.root, "splice"), exist_ok=True)

    # --- nominal features ------------------------------------------------
    def _device(self):
        import torch
        try:
            return next(self.pol._policy.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def nominal_features(self, scene: int, noise_seed: int, env_seed: int) -> list[dict]:
        """Forward 0..drive_until-1 features of the unperturbed episode, recorded once per
        (scene, noise seed, env seed) and cached on disk so a resumed run reuses them.

        Only the first `drive_until` forwards are needed, so the nominal episode is
        stepped only that far (the rest of it cannot influence those features)."""
        import torch
        from ...capture.splice import attach_recorder, attach_splice
        key = (int(scene), int(noise_seed), int(env_seed), self.drive_until)
        if key in self._nominal:
            return self._nominal[key]
        path = os.path.join(self.feature_cache, "s{}_n{}_e{}_k{}.pt".format(*key))
        dev = self._device()
        if os.path.exists(path):
            recs = [{k: v.to(dev) for k, v in r.items()} for r in torch.load(path, map_location="cpu")]
        else:
            env = self.make_env(scene)
            rec = attach_recorder(self.pol._policy)
            gen = attach_splice(self.pol._policy, source=lambda i: None, drive=None,
                                noise_key=noise_key_for(noise_seed))
            try:
                obs = env.reset(env_seed, PerturbationSpec())
                self.pol.reset()
                while len(rec.records) < self.drive_until:
                    a = self.pol(obs.policy_view())
                    if len(rec.records) >= self.drive_until:
                        break
                    obs, _succ, done, _ = env.step(a)
                    if done:
                        break
            finally:
                gen.detach(); rec.detach()
            recs = rec.records[: self.drive_until]
            if len(recs) < self.drive_until:
                raise RuntimeError(f"nominal episode of scene {scene} ended after {len(recs)} forwards")
            torch.save([{k: v.detach().cpu() for k, v in r.items()} for r in recs], path + ".part")
            os.replace(path + ".part", path)
        # one scene's features at a time on the GPU: R-047 OOM'd holding many
        self._nominal = {key: recs}
        return recs

    # --- one episode -------------------------------------------------------
    def mint(self, episode_id: str, scene: int, spec: PerturbationSpec, prov: Provenance) -> EpisodeRecord:
        import torch
        from ...capture.splice import attach_splice
        from ...runner import rollout
        self.heldout.refuse(scene, prov.dir_seed)                  # G4, before any compute
        if prov.privileged:
            raise ValueError("rescue minting never reads privileged state")
        recs = self.nominal_features(scene, prov.noise_seed, prov.env_seed)
        env = self.make_env(scene)
        img_dir = self.store.image_dir(episode_id)
        if os.path.isdir(img_dir):                                  # a crashed earlier attempt
            for f in os.listdir(img_dir):
                os.remove(os.path.join(img_dir, f))
        env.image_dir = img_dir
        h = attach_splice(self.pol._policy, source=lambda i, recs=recs: recs[i] if i < len(recs) else None,
                          drive=self.drive, noise_key=noise_key_for(prov.noise_seed),
                          drive_until=self.drive_until)
        try:
            r = rollout(env, self.pol, prov.env_seed, spec, video_dir=None)
        finally:
            h.detach()
        if self.traces is not None:
            self.traces.append(r)
        fe = list(getattr(self.pol, "forward_env_steps", []))
        driven_until_step = fe[self.drive_until] if len(fe) > self.drive_until else r.env_steps
        frames = frames_from_rollout(r, self.store, driven_until_step)
        # the executed forward-0 chunks on record, for the R-054 spot check
        if h.records:
            a0 = {arm: h.records[0]["action"][arm][0, :, :LIVE_DIMS].float().cpu().numpy()
                  for arm in ("P", "N") if arm in h.records[0]["action"]}
            d_pn = float(np.linalg.norm(a0["P"] - a0["N"])) if len(a0) == 2 else float("nan")
            np.savez_compressed(os.path.join(self.store.root, "splice", f"{episode_id}.npz"),
                                env_steps=np.asarray(fe, np.int32), d_pn_f0=np.float32(d_pn),
                                **{f"act_{k}_f0": v for k, v in a0.items()})
        else:
            d_pn = float("nan")
        frame0 = r.steps[0].obs_state if r.steps else {}
        rec = EpisodeRecord(
            episode_id=episode_id, provenance=prov, instruction=r.instruction, frames=frames,
            outcome={"success": bool(r.success), "termination": r.termination, "steps": r.env_steps,
                     "forwards": len(fe), "closest_approach_m": closest_to_target(r, env),
                     "failure_family": None if r.success else failure_family(r),
                     "d_pn_f0": d_pn, "driven_until_step": int(driven_until_step)},
            exportable=bool(r.success), rollout_id=r.rollout_id,
            image_meta={"cameras": sorted(frames[0].images) if frames else [],
                        "size": int(getattr(env, "obs_size", 0)), "orientation": "env_raw",
                        "format": "png"},
            fingerprint={**(r.fingerprint or {}), "splice": {"drive": self.drive, "drive_until": self.drive_until,
                                                              "noise_key": "1000*noise_seed+forward"},
                         "init_qpos": frame0.get("joint_pos")})
        self.store.write(rec)
        del env
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return rec
