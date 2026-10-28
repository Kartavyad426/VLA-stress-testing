"""r2s_common: real2sim code shared by res31 (scene building), implres31 (checks) and TaskGen (task creation).
User 10-28: "yes" to one shared module instead of three copies. Owner: res31; others import it read-only and suggest
changes via prim (code-ownership rule). Each piece is the best existing version, copied with attribution:
  detect    Detector (counting protocol, models loaded once), assign, head_noun   from implres31 common.py
  kinds     object kinds, size bounds, gripper limits                              from TaskGen core/kinds.py
  geometry  aabb, cavity, horizontal_gap, oriented_size, footprint_box             from TaskGen core/geometry.py
            collision_points, footprint, touches_border                            from implres31 common.py
            surface_dist                                                           from res31 scene_checks.py
  scene     SceneFrames (built scene posed at recorded frames, cameras, render), composites   from implres31 common.py
Equivalence with each source: experiments/real2sim/tests/test_r2s_common.py.

Import r2s_common BEFORE mujoco: under MUJOCO_GL=osmesa, triton (loaded by transformers with the detector) segfaults
when mujoco was imported first (SCENE_V1_MECHANISMS K5); this package imports triton first when it is installed.
Exploratory (10-28).
"""
try:
    import triton  # noqa: F401  (K5)
except ImportError:
    pass
import os
import sys

R2S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/real2sim
if R2S not in sys.path:
    sys.path.insert(0, R2S)
