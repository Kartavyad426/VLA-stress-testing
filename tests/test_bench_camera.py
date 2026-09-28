"""camera_bench_* knobs reproduce LIBERO-Plus's camera geometry exactly (D5).

For each view string, LIBERO-Plus builds its env from
"<scene>_view_{yaw}_{pitch}_{scale*100}_0_0_initstate_0.bddl"; ours applies the
matching bench knobs to a canonical-camera variant of the same scene. The
agentview pose in `sim.model` must agree to 1e-4.

CPU only. Needs the LIBERO-Plus stack:

  PYTHONPATH=$PWD/third_party/LIBERO-plus LIBERO_CONFIG_PATH=$PWD/third_party/libero-plus-config \
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa .venvs/libero-plus/bin/python -m pytest tests/test_bench_camera.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import pytest

from vla_harness.schema import PerturbationSpec

pytest.importorskip("libero.libero.envs.problems.libero_tabletop_manipulation",
                    reason="needs the LIBERO-Plus fork on PYTHONPATH")

# A Language-Instructions variant: canonical camera, initstate 0 (as R-047's scenes).
SCENE = 984
# (bench yaw, bench pitch, bench scale) -> the view string LIBERO-Plus parses.
# Negative yaw is written as LIBERO-Plus stores it (285 = -75).
CASES = [
    ((75, 0, 1.0), "75_0_100_0_0"),
    ((-75, 0, 1.0), "285_0_100_0_0"),
    ((0, 15, 1.0), "0_15_100_0_0"),
    ((0, 0, 2.0), "0_0_200_0_0"),
    ((30, 15, 1.0), "30_15_100_0_0"),
    ((0, 0, 1.37), "0_0_137_0_0"),
    ((75, 15, 2.0), "75_15_200_0_0"),
]


def _agentview(sim):
    cid = sim.model.camera_name2id("agentview")
    return np.array(sim.model.cam_pos[cid], float), np.array(sim.model.cam_quat[cid], float)


def _close_quat(a, b, tol):
    return min(np.abs(a - b).max(), np.abs(a + b).max()) <= tol     # q and -q are one rotation


@pytest.fixture(scope="module")
def env():
    from vla_harness.envs.libero_env import LiberoEnv
    e = LiberoEnv(suite="libero_spatial", task_id=SCENE, libero_plus=True, obs_size=64)
    e.reset(0, PerturbationSpec())
    return e


def _libero_plus_view(env, view):
    from libero.libero.envs import OffScreenRenderEnv
    base = env._env.unwrapped._task_bddl_file.split("_view_")[0]
    ref = OffScreenRenderEnv(bddl_file_name=f"{base}_view_{view}_initstate_0.bddl",
                             camera_heights=64, camera_widths=64)
    try:
        return _agentview(ref.env.sim)
    finally:
        ref.close()


@pytest.mark.parametrize("knobs,view", CASES, ids=[v for _, v in CASES])
def test_bench_knobs_match_libero_plus(env, knobs, view):
    yaw, pitch, scale = knobs
    env.reset(0, PerturbationSpec.of(camera_bench_yaw_deg=yaw, camera_bench_pitch_deg=pitch,
                                     camera_bench_scale=scale))
    pos, quat = _agentview(env._sim())
    ref_pos, ref_quat = _libero_plus_view(env, view)
    assert np.abs(pos - ref_pos).max() <= 1e-4, (pos, ref_pos)
    assert _close_quat(quat, ref_quat, 1e-4), (quat, ref_quat)


def test_neutral_bench_knobs_leave_camera_at_nominal(env):
    env.reset(0, PerturbationSpec())
    pos0, quat0 = _agentview(env._sim())
    env.reset(0, PerturbationSpec.of(camera_bench_yaw_deg=0, camera_bench_pitch_deg=0,
                                     camera_bench_scale=1.0))
    pos, quat = _agentview(env._sim())
    assert np.array_equal(pos, pos0) and np.array_equal(quat, quat0)


def test_mixing_camera_families_raises(env):
    with pytest.raises(ValueError, match="bench"):
        env.reset(0, PerturbationSpec.of(camera_yaw_deg=5.0, camera_bench_yaw_deg=5))


def test_scale_neutral_is_one():
    sp = PerturbationSpec.of(camera_bench_scale=1.5, camera_bench_yaw_deg=30)
    assert dict(sp.revert("camera_bench_scale").knobs)["camera_bench_scale"] == 1.0
    assert PerturbationSpec.of(camera_bench_scale=1.0).is_nominal()
    assert PerturbationSpec.of(camera_bench_scale=1.0).label() == "nominal"
