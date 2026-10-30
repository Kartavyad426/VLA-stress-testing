"""Shared pieces for implres31's REAL2SIM_CHECKLIST_V1 checks (A2, C1-C5, D1-D6): a built scene with the sim posed at
the recorded joints for the frames before anything is touched (checklist rule 3), the real frames at the same moments,
sim segmentation and depth, our composites, and one detector that follows the checklist's counting protocol.

res31's modules are imported read-only from the main checkout (--real2sim; uncommitted there). Nothing here writes into
a scene directory: composites need the scene's cached plates (plate_cache_lama_frame0_plate.png, plate_cache_wrist_frame0,
table_tex_wrist.npz, gripper_lib/), which build_v1 / scene_review make; a scene without them gets no composite checks.
Exploratory, UNVERIFIED.
"""
import json, os, sys
try:  # under MUJOCO_GL=osmesa, triton (loaded by transformers with the detector) segfaults if mujoco was loaded first (K5)
    import triton  # noqa: F401
except ImportError:
    pass
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import cv2
import h5py
import mujoco

MAIN = "/home/imerit/Documents/Code/VLA"
R2S = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--real2sim"), os.path.join(MAIN, "experiments/real2sim"))
sys.path.insert(0, R2S)
from episode_info import episode_info  # noqa: E402
from scene_roles import container_bodies, table_label  # noqa: E402
from composite import cam_K, cam_pose  # noqa: E402

ROBOT_PREFIX = ("fr3", "g_")
POLICY_HW = (180, 320)  # what GR00T-DROID sees (scene_review, policy_rollout)


def head_noun(label):
    """'green block#2' -> 'block' (same rule as block_size_check.head_noun; colour words from build_scene.COLOURS)."""
    from build_scene import COLOURS
    words = [w for w in label.split("#")[0].lower().split() if w not in COLOURS]
    return words[-1] if words else label.split("#")[0]


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


class Detector:
    """Grounding DINO tiny + SAM base, loaded once, with the checklist's counting protocol (REAL2SIM_CHECKLIST_V1 §2):
    one detector call per noun (no colour words), robot pixels masked, detections assigned to sim objects by mask overlap.
    Box filtering is perceive.detect_multi's (thresholds 0.30/0.25, boxes > 15% of the image dropped, group boxes dropped,
    NMS 0.5), re-implemented here only to keep the models loaded between calls."""

    def __init__(self, box_thr=0.30, text_thr=0.25, nms_iou=0.5, max_frac=0.15):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
        self.torch = torch
        self.dp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()
        self.sp = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.sm = SamModel.from_pretrained("facebook/sam-vit-base").eval()
        self.box_thr, self.text_thr, self.nms_iou, self.max_frac = box_thr, text_thr, nms_iou, max_frac
        self.calls = 0

    def boxes(self, img, noun):
        from PIL import Image
        from perceive import _iou, _contains
        H, W = img.shape[:2]
        inputs = self.dp(images=Image.fromarray(img), text=noun + ".", return_tensors="pt")
        with self.torch.no_grad():
            o = self.dm(**inputs)
        self.calls += 1
        r = self.dp.post_process_grounded_object_detection(o, inputs.input_ids, threshold=self.box_thr, text_threshold=self.text_thr,
                                                           target_sizes=[(H, W)])[0]
        c = [(s, b) for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()) if (b[2] - b[0]) * (b[3] - b[1]) <= self.max_frac * H * W]
        c = [x for x in c if sum(_contains(x[1], y[1]) for y in c if y is not x) < 2]
        c.sort(key=lambda x: -x[0])
        kept = []
        for x in c:
            if all(_iou(x[1], k[1]) < self.nms_iou for k in kept):
                kept.append(x)
        return kept

    def masks(self, img, noun, robot=None):
        """Instance masks (bool HxW) for `noun`, robot pixels removed; [] if none."""
        from PIL import Image
        kept = self.boxes(img, noun)
        if not kept:
            return []
        pil = Image.fromarray(img)
        inputs = self.sp(pil, input_boxes=[[b for _, b in kept]], return_tensors="pt")
        with self.torch.no_grad():
            o = self.sm(**inputs)
        ms = self.sp.image_processor.post_process_masks(o.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
        out = []
        for i in range(len(kept)):
            mk = ms[i][int(o.iou_scores[0, i].argmax())].numpy().astype(bool)
            if robot is not None:
                mk &= ~robot
            if mk.sum() > 0:
                out.append(mk)
        return out


def assign(cands, sim, window=60):
    """The candidate mask that best overlaps `sim` among those lying mostly inside sim's box grown by `window` px
    (block_size_check.real_by_mask's rule). Returns (mask, iou) or (None, 0)."""
    ys, xs = np.nonzero(sim)
    gate = np.zeros_like(sim)
    gate[max(ys.min() - window, 0):ys.max() + window, max(xs.min() - window, 0):xs.max() + window] = True
    best, score = None, 0.0
    for mk in cands:
        if (mk & gate).sum() < 0.5 * mk.sum():
            continue
        iou = (mk & sim).sum() / max((mk | sim).sum(), 1)
        if iou > score:
            best, score = mk, float(iou)
    return best, score


_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def collision_points(m, d, geoms):
    """World corners of the collision geoms' bounding boxes (N*8, 3); visual-only geoms are left out."""
    pts = [d.geom_xpos[g] + (m.geom_aabb[g][:3] + m.geom_aabb[g][3:] * _SIGNS) @ d.geom_xmat[g].reshape(3, 3).T
           for g in geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def footprint(m, d, geoms):
    """(centre xy, radius) of the horizontal bounding box of the collision geoms; bottom z."""
    P = collision_points(m, d, geoms)
    lo, hi = P.min(0), P.max(0)
    return (lo[:2] + hi[:2]) / 2, float(max(hi[0] - lo[0], hi[1] - lo[1]) / 2), float(lo[2])


def touches_border(mk):
    return bool(mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any())


def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)
