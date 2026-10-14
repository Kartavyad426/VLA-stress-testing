"""Real-to-sim step 1: perception on one frame of a DROID episode (CPU only).

Grounding DINO boxes -> SAM masks -> MoGe-2 metric point map + intrinsics.
Writes <out>/perception.npz and <out>/perception_check.png (masks drawn on the frame).

    .venvs/real2sim/bin/python experiments/real2sim/perceive.py \
        --episode data/real2sim/droid/ep_orange_block --cam 22008760 --out runs/real2sim/ep_orange_block
"""
import argparse, json, os
import numpy as np
import cv2
import torch
from PIL import Image

PROMPTS = ["black bowl", "orange block", "yellow block", "green block", "wooden table", "robot gripper"]


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    assert ok, f"cannot read frame {idx} of {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detect(img, prompts, box_thr=0.30, text_thr=0.25):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval()
    out = {}
    pil = Image.fromarray(img)
    for p in prompts:  # one prompt at a time: no label-merging ambiguity between similar phrases
        inputs = proc(images=pil, text=p + ".", return_tensors="pt")
        with torch.no_grad():
            o = model(**inputs)
        r = proc.post_process_grounded_object_detection(
            o, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
            target_sizes=[img.shape[:2]])[0]
        if len(r["scores"]) == 0:
            out[p] = None
            continue
        i = int(r["scores"].argmax())
        out[p] = dict(box=r["boxes"][i].tolist(), score=float(r["scores"][i]))
    return out


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / max(1e-6, area(a) + area(b) - inter)


def _contains(big, small, tol=4):
    return big[0] - tol <= small[0] and big[1] - tol <= small[1] and big[2] + tol >= small[2] and big[3] + tol >= small[3]


def detect_multi(img, prompts, box_thr=0.30, text_thr=0.25, nms_iou=0.5, max_frac=0.15):
    """Every instance per prompt (cluttered scenes). Instances across all these prompts are merged by NMS;
    boxes covering > max_frac of the image, or containing two or more other boxes (group boxes), are dropped.
    Returns {"<prompt>#<k>": {box, score}}."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval()
    pil = Image.fromarray(img)
    H, W = img.shape[:2]
    cands = []
    for p in prompts:
        inputs = proc(images=pil, text=p + ".", return_tensors="pt")
        with torch.no_grad():
            o = model(**inputs)
        r = proc.post_process_grounded_object_detection(
            o, inputs.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[img.shape[:2]])[0]
        for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()):
            if (b[2] - b[0]) * (b[3] - b[1]) <= max_frac * H * W:
                cands.append((s, p, b))
    cands = [c for c in cands if sum(_contains(c[2], d[2]) for d in cands if d is not c) < 2]
    cands.sort(key=lambda c: -c[0])
    kept = []
    for c in cands:
        if all(_iou(c[2], k[2]) < nms_iou for k in kept):
            kept.append(c)
    out, n = {}, {}
    for s, p, b in kept:
        n[p] = n.get(p, 0) + 1
        out[f"{p}#{n[p]}"] = dict(box=b, score=float(s))
    return out


def segment(img, boxes):
    from transformers import SamModel, SamProcessor
    mid = "facebook/sam-vit-base"
    proc = SamProcessor.from_pretrained(mid)
    model = SamModel.from_pretrained(mid).eval()
    masks = {}
    pil = Image.fromarray(img)
    for name, det in boxes.items():
        if det is None:
            continue
        inputs = proc(pil, input_boxes=[[det["box"]]], return_tensors="pt")
        with torch.no_grad():
            o = model(**inputs)
        m = proc.image_processor.post_process_masks(
            o.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0][0]
        best = int(o.iou_scores[0, 0].argmax())
        masks[name] = m[best].numpy().astype(bool)
    return masks


def geometry(img, fov_x_deg=None):
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").eval()
    x = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)
    with torch.no_grad():
        o = model.infer(x, fov_x=fov_x_deg, use_fp16=False)  # fp16 is GPU-only
    return {k: v.numpy() for k, v in o.items() if isinstance(v, torch.Tensor)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=",".join(PROMPTS), help="single-instance prompts (best box each), comma-separated")
    ap.add_argument("--multi", default="", help="multi-instance prompts for cluttered scenes, comma-separated")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    img = read_frame(os.path.join(a.episode, f"{a.cam}.mp4"), a.frame)
    cache = os.path.join(a.out, "masks_cache.npz")
    if os.path.exists(cache):  # detection + SAM are the slow CPU part; reuse them on reruns
        c = np.load(cache, allow_pickle=True)
        dets, masks = c["dets"].item(), c["masks"].item()
    else:
        dets = detect(img, [x.strip() for x in a.prompts.split(",") if x.strip()])
        if a.multi:
            dets.update(detect_multi(img, [x.strip() for x in a.multi.split(",") if x.strip()]))
        masks = segment(img, dets)
        np.savez_compressed(cache, dets=np.array(dets, dtype=object), masks=np.array(masks, dtype=object))
    geo = geometry(img)  # FOV left to MoGe: no ZED intrinsics without the SDK

    H, W = img.shape[:2]
    K = geo["intrinsics"].copy()  # MoGe returns normalised intrinsics
    K[0] *= W
    K[1] *= H
    np.savez_compressed(
        os.path.join(a.out, "perception.npz"), image=img, points=geo["points"],
        depth=geo["depth"], valid=geo["mask"], K=K,
        **{f"mask/{k}": v for k, v in masks.items()})
    json.dump(dict(cam=a.cam, frame=a.frame, detections=dets, K=K.tolist(),
                   fov_x_deg=float(np.degrees(2 * np.arctan(W / (2 * K[0, 0]))))),
              open(os.path.join(a.out, "perception.json"), "w"), indent=1)

    vis = img.copy()
    rng = np.random.default_rng(0)
    for name, m in masks.items():
        c = rng.integers(60, 255, 3)
        vis[m] = (0.5 * vis[m] + 0.5 * c).astype(np.uint8)
        x0, y0 = map(int, dets[name]["box"][:2])
        cv2.putText(vis, f"{name} {dets[name]['score']:.2f}", (x0, max(15, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(a.out, "perception_check.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(json.dumps({k: (None if v is None else round(v["score"], 3)) for k, v in dets.items()}))
    print("fov_x_deg", round(float(np.degrees(2 * np.arctan(W / (2 * K[0, 0])))), 1))


if __name__ == "__main__":
    main()
