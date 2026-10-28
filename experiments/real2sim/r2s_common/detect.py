"""The checklist's counting protocol (REAL2SIM_CHECKLIST_V1 §2): one detector call per noun (no colour words), robot
pixels masked, detections assigned to sim objects by mask overlap; models loaded ONCE.
Copied from implres31 experiments/real2sim_checks/common.py (worktree exp/real2sim-checks, 10-28): Detector, assign,
head_noun. Box filtering = perceive.detect_multi's thresholds (0.30 / 0.25, boxes > 15 % of the image dropped, group
boxes dropped, NMS 0.5). Exploratory, UNVERIFIED.
"""
import numpy as np
from . import R2S  # noqa: F401  (puts experiments/real2sim on sys.path for perceive / build_scene)


def head_noun(label):
    """'green block#2' -> 'block' (same rule as block_size_check.head_noun; colour words from build_scene.COLOURS)."""
    from build_scene import COLOURS
    words = [w for w in label.split("#")[0].lower().split() if w not in COLOURS]
    return words[-1] if words else label.split("#")[0]


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
