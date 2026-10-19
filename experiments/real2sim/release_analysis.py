"""Where does a policy rollout's place fail? Offline analysis of saved rollouts (policy_rollout.py rollout.npz), CPU.

  D  carry vs real: the policy's grasp->release displacement in sim vs the real operator's (recorded joints, FK) and
     vs the sim object->container offset. Tests "the policy moves real-world distances; our scene's are off".
  C  objects in the images: each sim object rendered in the exterior camera vs the real object's perception mask
     (frame 0) - centroid offset (px and cm at the object's depth) and IoU. Tests "the policy aims at a bowl drawn in
     the wrong place". Also reports whether the cleaned background still shows a dark bowl-like blob (ghost).
  B  the two heads of the DROID action: hand target (eef_9d -> position) vs FK of the joint targets (what we execute),
     per step and by position in the 40-step chunk; at the release step, where each would put the fingertip point.

    MUJOCO_GL=egl .venvs/real2sim/bin/python experiments/real2sim/release_analysis.py \
        --scene runs/real2sim/ep_orange_block_stereo --episode data/real2sim/droid/ep_orange_block \
        --rollouts runs/real2sim/policy/g2_green runs/real2sim/policy/g2_orange
Writes <first rollout>/../release_analysis.json and prints a summary.
"""
import argparse, json, os, sys
import numpy as np
import cv2
import mujoco
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_scene import FLANGE_IN_LINK7  # noqa: E402
from composite import cam_K, cam_pose  # noqa: E402

OPEN_LOOP = 15  # policy_rollout default: steps executed per chunk


def segments(mask):
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            e = k + (int(np.argmax(~mask[k:])) if (~mask[k:]).any() else len(mask) - k)
            out.append((k, e))
            k = e
        else:
            k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--lift_m", type=float, default=0.02)
    a = ap.parse_args()
    m = mujoco.MjModel.from_binary_path(os.path.join(a.scene, "scene.mjb"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    arm = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
    l7, pinch = m.body("fr3_link7").id, m.site("g_pinch").id
    res = dict(scene=a.scene, episode=a.episode)

    # real operator: grasp/release fingertip points (FK of recorded joints) in this scene's robot frame
    h = h5py.File(os.path.join(a.episode, "trajectory.h5"))
    J, G = h["observation/robot_state/joint_positions"][:], h["observation/robot_state/gripper_position"][:]
    dk = mujoco.MjData(m)

    def fk(q):
        dk.qpos[:] = m.qpos0
        dk.qpos[arm] = q
        mujoco.mj_kinematics(m, dk)
        return dk.site_xpos[pinch].copy(), dk.xpos[l7] + dk.xmat[l7].reshape(3, 3) @ FLANGE_IN_LINK7, dk.xmat[l7].reshape(3, 3).copy()
    kc = int(np.argmax(G > 0.5))
    ko = kc + int(np.argmax(G[kc:] < 0.5))
    rg, rr = fk(J[kc])[0], fk(J[ko])[0]
    res["real"] = dict(close_frame=kc, open_frame=ko, carry_xy_m=(rr - rg)[:2].tolist(), carry_dist_m=float(np.linalg.norm((rr - rg)[:2])),
                       release_height_m=float(rr[2]))

    # D: per rollout, the successful carry (closed + target lifted) and where the policy opened
    bowl = d.xpos[m.body("black_bowl").id].copy()
    res["bowl_xyz"] = bowl.tolist()
    res["rollouts"] = {}
    for rd in a.rollouts:
        z = np.load(os.path.join(rd, "rollout.npz"))
        rj = json.load(open(os.path.join(rd, "rollout.json")))
        tb = m.body(rj["task"]["target"]).id
        pa, pc, tp, g, act = z["pinch_ach"], z["pinch_cmd"], z["target_pos"], z["grip_cmd"], z["action"]
        lifted = tp[:, 2] - tp[0, 2] >= a.lift_m
        rows = []
        for s, e in segments(g):
            if not lifted[s:e].any():
                continue
            k_rel = min(e, len(g) - 1)  # first step commanded open
            grasp_pt, rel_pt = pa[s], pa[k_rel]
            obj0 = tp[s]
            # B at the release step: hand head vs joint head (as fingertip points, same orientation offset)
            hand_flange = act[k_rel, :3]
            _, jflange, R = fk(act[k_rel, 10:17])
            pinch_off = pc[k_rel] - jflange  # flange -> pinch offset of the commanded pose
            hand_pinch = hand_flange + pinch_off
            rows.append(dict(close_step=s, release_step=k_rel, carry_xy_m=(rel_pt - grasp_pt)[:2].tolist(),
                             carry_dist_m=float(np.linalg.norm((rel_pt - grasp_pt)[:2])),
                             object_to_bowl_xy_m=(bowl - obj0)[:2].tolist(), object_to_bowl_dist_m=float(np.linalg.norm((bowl - obj0)[:2])),
                             release_to_bowl_m=float(np.linalg.norm((rel_pt - bowl)[:2])), release_height_above_bowl_base_m=float(rel_pt[2] - bowl[2]),
                             B_hand_head_pinch_to_bowl_m=float(np.linalg.norm((hand_pinch - bowl)[:2])),
                             B_joint_head_pinch_to_bowl_m=float(np.linalg.norm((pc[k_rel] - bowl)[:2])),
                             B_heads_diff_m=float(np.linalg.norm(hand_flange - jflange)),
                             chunk_index_at_release=int(k_rel % OPEN_LOOP)))
        # B over the whole rollout, by index within the executed part of the chunk
        diff = np.array([np.linalg.norm(act[k, :3] - fk(act[k, 10:17])[1]) for k in range(len(act))])
        by_idx = {i: float(np.median(diff[i::OPEN_LOOP])) for i in range(OPEN_LOOP)}
        res["rollouts"][os.path.basename(rd)] = dict(instruction=rj["instruction"], carries=rows,
                                                     B_heads_diff_median_m=float(np.median(diff)), B_heads_diff_by_chunk_index_m=by_idx)

    # C: sim objects vs real objects in the exterior image (frame 0)
    per = dict(np.load(os.path.join(a.scene, "perception.npz")))
    H, W = per["image"].shape[:2]
    cid = m.camera("ext").id
    K = cam_K(m, cid, W, H)
    R, t = cam_pose(d, cid)
    r = mujoco.Renderer(m, H, W)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera="ext")
    seg = r.render()
    r.close()
    cobj = {}
    for name, o in json.load(open(os.path.join(a.scene, "scene_fit.json")))["objects"].items():
        if "body" not in o or f"mask/{name}" not in per:
            continue
        b = m.body(o["body"]).id
        sim = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == b])
        real = per[f"mask/{name}"].astype(bool)
        if not sim.any():
            continue
        cs, cr = np.argwhere(sim).mean(0)[::-1], np.argwhere(real).mean(0)[::-1]
        depth = float(((d.xpos[b] - t) @ R)[2])
        px = float(np.linalg.norm(cs - cr))
        cobj[name] = dict(iou=float((sim & real).sum() / (sim | real).sum()), centroid_offset_px=px,
                          centroid_offset_cm_at_depth=100 * px * depth / K[0, 0], sim_px=int(sim.sum()), real_px=int(real.sum()))
    res["C_objects_in_exterior_image"] = cobj
    out = os.path.join(os.path.dirname(os.path.normpath(a.rollouts[0])), "release_analysis.json")
    json.dump(res, open(out, "w"), indent=1)

    print(f"REAL operator: carry {res['real']['carry_dist_m']*100:.1f} cm (grasp f{kc} -> release f{ko})")
    for k, v in res["rollouts"].items():
        print(f"{k} [{v['instruction']}]: heads differ median {v['B_heads_diff_median_m']*100:.1f} cm; by chunk index (cm): "
              + " ".join(f"{i}:{x*100:.1f}" for i, x in v["B_heads_diff_by_chunk_index_m"].items()))
        for c in v["carries"]:
            print(f"  carry steps {c['close_step']}->{c['release_step']}: policy carried {c['carry_dist_m']*100:.1f} cm; object->bowl {c['object_to_bowl_dist_m']*100:.1f} cm; "
                  f"released {c['release_to_bowl_m']*100:.1f} cm from bowl centre, {c['release_height_above_bowl_base_m']*100:.1f} cm above its base; "
                  f"hand head would be {c['B_hand_head_pinch_to_bowl_m']*100:.1f} cm, joint head {c['B_joint_head_pinch_to_bowl_m']*100:.1f} cm (heads {c['B_heads_diff_m']*100:.1f} cm apart, chunk idx {c['chunk_index_at_release']})")
    for k, v in cobj.items():
        print(f"C {k}: IoU sim vs real {v['iou']:.2f}, centroid offset {v['centroid_offset_px']:.1f} px = {v['centroid_offset_cm_at_depth']:.1f} cm")
    print("wrote", out)


if __name__ == "__main__":
    main()
