"""Human demo loader for the DMP PoC (docs/DMP_POC_METHOD.html §1.1, §4, §7 T0-lite). CPU.

For each of the 500 LIBERO-Spatial demos in <main>/data/libero_raw/libero_spatial,
and each step: EEF position and rotation, gripper command and aperture, and the
positions of the target bowl and the plate. Poses come from setting the sim to
the recorded state and reading the bodies and the grip site (no stepping), so
they are the sim's, in the same frames as the R-047 logs.

Segments split at gripper-command changes (1-2 step blips ignored). The
retargeted phases are defined around the FINAL grasp, the last closed run that
lifts the bowl >= LIFT_OK_M and is followed by a release:
  A approach  the open run just before the final grasp
  B carry     the final grasp, until the release command
  C retreat   everything after the release
Earlier grasp cycles are prior attempts and are tagged, not used as phases:
  regrasp         an earlier close lifted the bowl < LIFT_FAIL_M (failed grasp,
                  then a second approach: A starts from a post-failure state)
  replace         an earlier close lifted the bowl >= LIFT_OK_M (carried, set
                  down, picked up again)
  trailing_close  a closed run after the final release (ignored)
A demo without a final grasp followed by a release is not usable (user 10-14:
the 6 never-release demos and the 10 T0-lite failures stay dropped; the 39
multi-attempt demos are kept, tagged).

Object frames are translation-only (bowl and plate are rotationally symmetric):
a goal is the demo's EEF-minus-object offset placed at the object's current
position, with the demo's world orientation.

T0-lite: the demo's last recorded state must pass the env's success check.

Cached per task to runs/dmp_poc/demos/<task>.npz.

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restore import EEF_SITE, WORKTREE, main_repo  # noqa: E402

RAW = os.path.join(main_repo(), "data/libero_raw/libero_spatial")
BDDL = os.path.join(main_repo(), "third_party/LIBERO-plus/libero/libero/bddl_files/libero_spatial")
CACHE = os.path.join(WORKTREE, "runs/dmp_poc/demos")


def task_name(instruction: str) -> str:
    return instruction.strip().lower().replace(" ", "_")


def demo_file(task: str) -> str:
    return os.path.join(RAW, f"{task}_demo.hdf5")


def tasks():
    return sorted(os.path.basename(p)[:-len("_demo.hdf5")] for p in glob.glob(os.path.join(RAW, "*_demo.hdf5")))


def control_env(task: str):
    from libero.libero.envs.env_wrapper import ControlEnv
    env = ControlEnv(bddl_file_name=os.path.join(BDDL, task + ".bddl"),
                     has_offscreen_renderer=False, use_camera_obs=False)
    env.reset()
    return env


BLIP_STEPS = 2       # gripper-command runs this short are operator blips, not events
LIFT_OK_M = 0.05     # a grasp that lifts the bowl this far held it
LIFT_FAIL_M = 0.015  # a grasp that lifts it less than this missed (ROOT_IDENTIFICATION miss/slip)


def debounce(grip_cmd: np.ndarray, blip=BLIP_STEPS) -> np.ndarray:
    """Sign of the gripper command with interior runs of <= blip steps absorbed
    into the preceding run."""
    sign = np.sign(grip_cmd).astype(int)
    sign[sign == 0] = -1
    out = sign.copy()
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and sign[j + 1] == sign[i]:
            j += 1
        if i > 0 and j + 1 < len(out) and (j - i + 1) <= blip:
            out[i:j + 1] = out[i - 1]
        i = j + 1
    return out


def segments(grip_cmd: np.ndarray):
    """Boundaries [(start, end_exclusive), ...] at gripper command changes,
    after debouncing blips."""
    sign = debounce(grip_cmd)
    cuts = [0] + [i for i in range(1, len(sign)) if sign[i] != sign[i - 1]] + [len(sign)]
    return list(zip(cuts[:-1], cuts[1:]))


@dataclass
class Demo:
    task: str
    idx: int
    eef_pos: np.ndarray         # (T, 3)
    eef_quat: np.ndarray        # (T, 4) scipy x, y, z, w
    grip_cmd: np.ndarray        # (T,)  demo action, -1 open / +1 close
    aperture: np.ndarray        # (T,)  finger gap, m
    target_pos: np.ndarray      # (T, 3) target bowl
    plate_pos: np.ndarray       # (T, 3)
    segs: list                  # [(start, end), ...]
    clean_last_state: bool      # T0-lite

    @property
    def canonical(self) -> bool:
        """Approach, one grasp, one release: segments open / close / open."""
        return len(self.segs) == 3 and self.grip_cmd[self.segs[1][0]] > 0

    def lift(self, s, e):
        return float(self.target_pos[s:e, 2].max() - self.target_pos[0, 2])

    @property
    def phases(self):
        """(A, B, C) as (start, end) step ranges, or None if no final grasp with a
        release. Cached."""
        if not hasattr(self, "_phases"):
            closed = [k for k, (s, e) in enumerate(self.segs) if self.grip_cmd[s] > 0]
            final = [k for k in closed if self.lift(*self.segs[k]) >= LIFT_OK_M and k + 1 < len(self.segs) and k > 0]
            if not final:
                self._phases = None
            else:
                k = final[-1]
                self._phases = (self.segs[k - 1], self.segs[k], (self.segs[k + 1][0], self.segs[-1][1]))
                self._final_k = k
        return self._phases

    @property
    def tags(self) -> list[str]:
        if self.phases is None:
            return []
        k = self._final_k
        prior = [j for j in range(k) if self.grip_cmd[self.segs[j][0]] > 0]
        out = []
        if any(self.lift(*self.segs[j]) < LIFT_FAIL_M for j in prior):
            out.append("regrasp")
        if any(self.lift(*self.segs[j]) >= LIFT_OK_M for j in prior):
            out.append("replace")
        if prior and not out:
            out.append("partial_lift")      # an earlier close lifted it between the two thresholds
        if any(self.grip_cmd[s] > 0 for s, e in self.segs[k + 1:]):
            out.append("trailing_close")
        return out

    @property
    def usable(self) -> bool:
        return self.clean_last_state and self.phases is not None

    @property
    def eef_rot(self) -> Rotation:
        return Rotation.from_quat(self.eef_quat)

    def segment(self, k):
        s, e = self.segs[k]
        return s, e


def _read_task(task: str):
    env = control_env(task)
    sim = env.sim
    sid = sim.model.site_name2id(EEF_SITE)
    ooi = env.obj_of_interest
    tgt = [o for o in ooi if "bowl" in o][0]
    plate = [o for o in ooi if "plate" in o][0]
    bt, bp = sim.model.body_name2id(tgt + "_main"), sim.model.body_name2id(plate + "_main")
    out = []
    with h5py.File(demo_file(task), "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            g = f["data"][k]
            S, A = g["states"][:], g["actions"][:]
            T = len(S)
            ep, eq, tp, pp = np.zeros((T, 3)), np.zeros((T, 4)), np.zeros((T, 3)), np.zeros((T, 3))
            for t in range(T):
                env.set_state(S[t])
                sim.forward()
                ep[t] = sim.data.site_xpos[sid]
                eq[t] = Rotation.from_matrix(np.array(sim.data.site_xmat[sid]).reshape(3, 3)).as_quat()
                tp[t] = sim.data.body_xpos[bt]
                pp[t] = sim.data.body_xpos[bp]
            ok = bool(env.check_success())          # state is the last one here
            gs = g["obs/gripper_states"][:]
            out.append(dict(idx=int(k.split("_")[1]), eef_pos=ep, eef_quat=eq, grip_cmd=A[:, 6].copy(),
                            aperture=gs[:, 0] - gs[:, 1], target_pos=tp, plate_pos=pp, clean=ok))
    env.close()
    return tgt, plate, out


def build(task: str):
    os.makedirs(CACHE, exist_ok=True)
    tgt, plate, demos = _read_task(task)
    arrs = {"target": tgt, "plate": plate, "n": len(demos)}
    for d in demos:
        for key, v in d.items():
            arrs[f"{d['idx']}/{key}"] = v
    np.savez_compressed(os.path.join(CACHE, task + ".npz"), **arrs)
    return load(task)


def load(task: str) -> list[Demo]:
    p = os.path.join(CACHE, task + ".npz")
    if not os.path.exists(p):
        return build(task)
    z = np.load(p, allow_pickle=False)
    idxs = sorted({int(k.split("/")[0]) for k in z.files if "/" in k})
    out = []
    for i in idxs:
        g = lambda key: z[f"{i}/{key}"]  # noqa: E731
        cmd = g("grip_cmd")
        out.append(Demo(task=task, idx=i, eef_pos=g("eef_pos"), eef_quat=g("eef_quat"), grip_cmd=cmd,
                        aperture=g("aperture"), target_pos=g("target_pos"), plate_pos=g("plate_pos"),
                        segs=segments(cmd), clean_last_state=bool(g("clean"))))
    return out


def t0_lite():
    rows = {}
    for task in tasks():
        ds = load(task)
        nseg = [len(d.segs) for d in ds]
        rows[task] = {"n": len(ds), "clean_last_state": int(sum(d.clean_last_state for d in ds)),
                      "canonical": int(sum(d.canonical for d in ds)), "usable": int(sum(d.usable for d in ds)),
                      "tags": {g: int(sum(g in d.tags for d in ds if d.usable))
                               for g in ("regrasp", "replace", "partial_lift", "trailing_close")},
                      "segments_min": min(nseg), "segments_max": max(nseg),
                      "steps_min": min(len(d.eef_pos) for d in ds), "steps_max": max(len(d.eef_pos) for d in ds)}
    return rows


if __name__ == "__main__":
    print(json.dumps(t0_lite(), indent=1))
