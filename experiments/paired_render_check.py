"""R-039 precondition: can a LIBERO-Plus vision variant be re-rendered at the
NOMINAL condition mid-episode without stepping physics?  (Paired rendering.)

Run in the libero-plus venv with the fork on PYTHONPATH:
  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config \\
  MUJOCO_GL=egl .venvs/libero-plus/bin/python experiments/paired_render_check.py [out_dir]

Verified 2026-09-23 (RESULTS.md R-039, precondition 3): view 19.5 / light 44.3
mean-abs-pixel change, restore 0.001, physics unchanged; noise clean frame is
the inner env observation before the wrapper blurs it.

Original docstring: Step-1 check for R-039: can a LIBERO-Plus vision variant be re-rendered at the
NOMINAL condition mid-episode without stepping physics?

For each of view / light / noise: build the variant, reset, grab the perturbed
frame; write the nominal camera / light fields into sim.model (or read the
pre-blur frame); force-update observables; grab the nominal frame; restore;
grab again. Asserts qpos, qvel and sim time are unchanged throughout.
"""
import os, sys, numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/paired_render_check"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.getcwd())
from vla_harness.envs.libero_env import LiberoEnv
from vla_harness.schema import PerturbationSpec

# nominal agentview for the TABLETOP domain (libero_tabletop_manipulation.py:306-312)
POS_AV = np.array([0.6586131746834771, 0.0, 1.6103500240372423])
QUAT_AV = np.array([0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349])
# nominal lights (libero_tabletop_base_style.xml:45-46)
LIGHT_DIFFUSE = np.array([.8, .8, .8]); LIGHT_DIR = np.array([0, -.15, -1.]); LIGHT_SPEC = np.array([.3, .3, .3])

TASKS = {"view": 616, "noise": 1380, "light": 2113}

def phys(sim):
    return (sim.data.qpos.copy(), sim.data.qvel.copy(), float(sim.data.time))

def frame(env):
    inner = env._env.unwrapped._env            # LIBERO-plus ControlEnv wrapper
    return inner.env._get_observations(force_update=True)["agentview_image"].copy()

def wrapped_frame(env):
    inner = env._env.unwrapped._env
    obs = inner.env._get_observations(force_update=True)
    # the wrapper only blurs inside its own step()/reset(); replicate its post-hoc path
    return obs["agentview_image"].copy()

out = {}
for kind, tid in TASKS.items():
    env = LiberoEnv(suite="libero_spatial", task_id=tid, libero_plus=True,
                    libero_plus_base_instruction=True)
    obs0 = env.reset(0, PerturbationSpec())
    sim = env._sim(); wrapper = env._env.unwrapped._env
    p0 = phys(sim)
    pert = frame(env)
    cid = sim.model.camera_name2id("agentview")
    saved = dict(pos=sim.model.cam_pos[cid].copy(), quat=sim.model.cam_quat[cid].copy(),
                 diff=sim.model.light_diffuse.copy(), dir=sim.model.light_dir.copy(),
                 spec=sim.model.light_specular.copy())
    xpos_before = sim.data.cam_xpos[cid].copy()
    if kind == "view":
        sim.model.cam_pos[cid] = POS_AV; sim.model.cam_quat[cid] = QUAT_AV
    elif kind == "light":
        for i in range(sim.model.nlight):
            sim.model.light_diffuse[i] = LIGHT_DIFFUSE; sim.model.light_dir[i] = LIGHT_DIR
            sim.model.light_specular[i] = LIGHT_SPEC
    readback = dict(cam_pos=sim.model.cam_pos[cid].round(4).tolist(), diffuse=sim.model.light_diffuse[0].round(3).tolist())
    sim.forward()                                   # recompute data.cam_xpos / light_xpos from model
    xpos_after = sim.data.cam_xpos[cid].copy()
    nominal = frame(env)
    p1 = phys(sim)
    # restore and re-render: must reproduce the perturbed frame exactly
    sim.model.cam_pos[cid] = saved["pos"]; sim.model.cam_quat[cid] = saved["quat"]
    sim.model.light_diffuse[:] = saved["diff"]; sim.model.light_dir[:] = saved["dir"]
    sim.model.light_specular[:] = saved["spec"]
    sim.forward()
    again = frame(env)
    p2 = phys(sim)
    # the harness method must reproduce the manual nominal render
    via_env = env.nominal_frames()["image"]
    p3 = phys(sim)
    after_env = frame(env)                      # and leave the perturbed render intact
    d_env_vs_manual = float(np.abs(via_env.astype(int) - nominal.astype(int)).mean())
    d_env_restore = float(np.abs(after_env.astype(int) - pert.astype(int)).mean())
    # what the POLICY saw at reset (post-wrapper, i.e. post-blur for noise)
    policy_img = obs0.frames["image"]
    d_pn = float(np.abs(pert.astype(int) - nominal.astype(int)).mean())
    d_pa = float(np.abs(pert.astype(int) - again.astype(int)).mean())
    d_policy_vs_raw = float(np.abs(policy_img.astype(int)[::-1] - pert.astype(int)).mean()) if policy_img.shape == pert.shape else -1
    same_phys = (all(np.array_equal(a, b) for a, b in zip(p0[:2], p1[:2])) and all(np.array_equal(a, b) for a, b in zip(p0[:2], p2[:2]))
                 and all(np.array_equal(a, b) for a, b in zip(p0[:2], p3[:2])) and p0[2] == p1[2] == p2[2] == p3[2])
    out[kind] = dict(task=tid, noise=getattr(wrapper, "noise", None),
                     saved_cam_pos=saved["pos"].round(4).tolist(),
                     mean_abs_diff_pert_vs_nominal=d_pn, mean_abs_diff_pert_vs_restored=d_pa,
                     policy_frame_vs_raw_render=d_policy_vs_raw, physics_unchanged=same_phys,
                     env_nominal_frames_vs_manual=d_env_vs_manual, env_nominal_frames_restore=d_env_restore,
                     nlight=int(sim.model.nlight), img_shape=list(pert.shape),
                     readback=readback, cam_xpos_before=xpos_before.round(4).tolist(), cam_xpos_after=xpos_after.round(4).tolist())
    np.save(f"{OUT}/{kind}_pert.npy", pert); np.save(f"{OUT}/{kind}_nominal.npy", nominal)
    print(kind, out[kind], flush=True)
    env.close() if hasattr(env, "close") else None
import json; json.dump(out, open(f"{OUT}/paired_render.json", "w"), indent=1)
