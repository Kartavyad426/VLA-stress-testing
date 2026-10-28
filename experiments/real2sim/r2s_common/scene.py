"""A built scene posed at recorded frames: cameras, projection, segmentation and depth renders, composites.
Copied from implres31 experiments/real2sim_checks/common.py (worktree exp/real2sim-checks, 10-28): SceneFrames,
composites, ROBOT_PREFIX, POLICY_HW. Replaces res31's per-script loaders (geom_probe.Scene, camera_search.Search,
inspect_scene's camera mode) over time. Exploratory, UNVERIFIED.
"""
import json, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import cv2
import mujoco
from . import R2S  # noqa: F401
from episode_info import episode_info  # noqa: E402
from scene_roles import container_bodies, table_label  # noqa: E402
from composite import cam_K, cam_pose  # noqa: E402

ROBOT_PREFIX = ("fr3", "g_")
POLICY_HW = (180, 320)  # what GR00T-DROID sees (scene_review, policy_rollout)


class SceneFrames:
    """A built scene posed at recorded frames. frames(): the pre-touch frame set (frame 0 + episode_info 'before_touch')."""

    def __init__(self, scene, episode):
        self.scene, self.episode = scene, episode
        self.m = m = mujoco.MjModel.from_binary_path(os.path.join(scene, "scene.mjb"))
        self.fit = json.load(open(os.path.join(scene, "scene_fit.json")))
        self.per = json.load(open(os.path.join(scene, "perception.json")))
        self.st = np.load(os.path.join(scene, "replay_states.npz"))
        self.info = episode_info(episode)
        self.cams = {"ext": self.per["cam"], "wrist": self.fit["wrist_camera"]["serial"]}
        self.robot = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(ROBOT_PREFIX)])
        self.table = m.geom("table").id
        self.free = [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.containers = container_bodies(m, self.fit)
        # body name -> label in scene_fit (for nouns and masks)
        self.label = {o.get("body", lab.replace("#", "_").replace(" ", "_")): lab for lab, o in self.fit["objects"].items()}
        self.obj_geoms = {b: [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == b] for b in self.free}
        self.d = mujoco.MjData(m)
        self._caps = {}
        self._rend = {}

    def frames(self):
        return sorted({0, *self.info["frames"]["before_touch"]})

    def pose(self, frame):
        """Pose the sim at the replayed state nearest to real frame `frame`; returns the real frame index used."""
        k = int(np.argmin(np.abs(self.st["real_frames"] - frame)))
        self.d.qpos[:] = self.st["qpos"][k]
        mujoco.mj_forward(self.m, self.d)
        return int(self.st["real_frames"][k])

    def real(self, cam, frame, hw=None):
        """Real RGB frame (uint8) of camera 'ext' / 'wrist', area-resized to hw=(H, W) like the policy's input."""
        ser = self.cams[cam]
        if ser not in self._caps:
            self._caps[ser] = cv2.VideoCapture(os.path.join(self.episode, f"{ser}.mp4"))
        cap = self._caps[ser]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read {cam} frame {frame}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb if hw is None else cv2.resize(rgb, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)

    def render(self, cam, hw):
        """(geom id per pixel or -1, depth in m) of the current pose."""
        if hw not in self._rend:
            self._rend[hw] = mujoco.Renderer(self.m, *hw)
        r = self._rend[hw]
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        r.enable_segmentation_rendering()
        r.update_scene(self.d, camera=cam, scene_option=opt)
        seg = r.render()
        r.disable_segmentation_rendering()
        r.enable_depth_rendering()
        r.update_scene(self.d, camera=cam, scene_option=opt)
        depth = r.render().copy()
        r.disable_depth_rendering()
        geom = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        return geom, depth

    def K(self, cam, hw):
        return cam_K(self.m, self.m.camera(cam).id, hw[1], hw[0])

    def cam_pose(self, cam):
        return cam_pose(self.d, self.m.camera(cam).id)

    def project(self, cam, hw, P):
        """World points (N,3) -> pixel (N,2) and camera depth (N,)."""
        R, t = self.cam_pose(cam)
        pc = (np.asarray(P) - t) @ R
        K = self.K(cam, hw)
        z = pc[:, 2]
        uv = np.c_[pc[:, 0] / z * K[0, 0] + K[0, 2], pc[:, 1] / z * K[1, 1] + K[1, 2]]
        return uv, z

    def table_label(self):
        return table_label(self.per)


def composites(sf, hw=POLICY_HW):
    """(ext_fn, wrist_fn, note): functions of sf.d returning float32 RGB composites at hw, built exactly like
    scene_review --wrist v2 --wrist_blur gripper, from the scene's caches only. (None, None, reason) when a cache is missing."""
    from composite import Compositor, WristRenderV2, load_plate, wrist_plate, wrist_table_texture, load_gripper_library
    s = sf.scene
    need = ["plate_cache_lama_frame0_plate.png", "plate_cache_wrist_frame0.png", "plate_cache_wrist_frame0.json",
            "table_tex_wrist.npz", "gripper_lib/gripper_library.npz", "light_wrist.json"]
    miss = [n for n in need if not os.path.exists(os.path.join(s, n))]
    if miss:
        return None, None, "composite caches missing (built by scene_review / build_v1): " + ", ".join(miss)
    plate = load_plate(s, "lama", "frame0+plate")  # reads the cache; never computes (cache present)
    comp = Compositor(sf.m, s, plate, size=hw, verbose=False)
    wp = wrist_plate(sf.m, s, sf.episode, comp, sf.cams["wrist"])
    tt = wrist_table_texture(sf.m, s, sf.episode, sf.cams["wrist"], [])  # cache hit
    wc = WristRenderV2(sf.m, plate, comp, hw, wp, tt, load_gripper_library(os.path.join(s, "gripper_lib", "gripper_library.npz")),
                       light=json.load(open(os.path.join(s, "light_wrist.json"))), near="gripper")
    return comp, wc, "scene_review --wrist v2 --wrist_blur gripper"
