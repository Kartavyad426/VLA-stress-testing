"""`_gt_scene_object_pos`: every FREE-JOINT body, fixtures excluded.

Verified once against the real LIBERO-Plus sim (5115b970e766: the ramekin moves
0.116 m and appears here; cabinet and stove do not). This stand-in pins the
filter -- free joints only, names via the joint's body -- so a refactor cannot
quietly widen it to fixtures or narrow it back to the BDDL task objects.
"""
from __future__ import annotations

import numpy as np

from vla_harness.envs.libero_env import LiberoEnv


class _Model:
    # joints: 0 free (bowl), 1 hinge (drawer), 2 free (ramekin), 3 slide (gripper)
    jnt_type = np.array([0, 3, 0, 2])
    jnt_bodyid = np.array([1, 2, 3, 4])
    njnt = 4
    _names = {1: "bowl_main", 2: "drawer_link", 3: "ramekin_main", 4: "finger"}

    def body_id2name(self, i):
        return self._names.get(int(i))


class _Data:
    body_xpos = np.arange(15, dtype=float).reshape(5, 3)


class _Sim:
    model = _Model()
    data = _Data()


def test_only_free_bodies_with_their_positions():
    env = LiberoEnv.__new__(LiberoEnv)          # no simulator needed
    got = env._scene_object_pos(_Sim())
    assert set(got) == {"bowl_main", "ramekin_main"}
    assert got["ramekin_main"] == [9.0, 10.0, 11.0]


def test_cache_follows_the_model():
    env = LiberoEnv.__new__(LiberoEnv)
    a, b = _Sim(), _Sim()
    b.model = _Model()
    b.model._names = {1: "cup_main", 3: "plate_main"}
    assert set(env._scene_object_pos(a)) == {"bowl_main", "ramekin_main"}
    assert set(env._scene_object_pos(b)) == {"cup_main", "plate_main"}
