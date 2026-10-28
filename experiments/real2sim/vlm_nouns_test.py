"""VLM noun-only object lists (SceneGen_v1_gaps_1028 row 5; user 10-28: "test if vlm will return noun only names and how
good they are").

Asks hosted VLMs (NVIDIA NIM) for the objects on the table as ONE generic noun per kind plus a count (colour and
material go in a separate field, never into the noun), on exterior frame 0 of each scene. Scores against a
reference list read by eye (res31, 10-28), then runs the scene's detector (Grounding DINO, one call per noun, as the
counting protocol) on each returned noun and counts what it finds.

    set -a; . ~/Documents/Code/robotics_agentic/.env; set +a
    runs/real2sim/v1_dev/py_triton_first.sh experiments/real2sim/vlm_nouns_test.py --out runs/real2sim/v1_dev/vlm_nouns
Writes <out>/vlm_nouns.json and <out>/<scene>_<model>_det.jpg. Exploratory, UNVERIFIED (reference lists by eye).
"""
import argparse, base64, io, json, os, re, sys, time, urllib.request, urllib.error
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perceive import read_frame  # noqa: E402

PROMPT = ("List the separate physical objects resting on the main table or cart surface in this image. "
          "Exclude the robot arm and gripper, the table itself, tape on the table, and anything on other furniture "
          "or in the background. For each KIND of object give ONE short generic noun that an object detector would "
          "understand (for example: block, bowl, cup, apple, marker), with NO colour, material or size words in the "
          "noun; put colours and other details in \"details\". Count identical kinds together. Answer ONLY with JSON: "
          '{"objects": [{"noun": "<one generic noun>", "count": <int>, "details": "<colours, material, shape>"}]}')

# reference lists, exterior camera 22008760 frame 0, main table only (res31 by eye, 10-28)
SCENES = {
    "orange": ("data/real2sim/droid/ep_orange_block", {"bowl": 1, "block": 3}),
    "c11": ("data/real2sim/droid/ep_c11", {"bowl": 1, "block": 2}),
    "c1b": ("data/real2sim/droid/ep_c1b", {"bowl": 1, "block": 2, "cylinder": 1, "cup": 2, "peg stand": 2}),
    "c3": ("data/real2sim/droid/ep_c3", {"bowl": 1, "block": 8, "apple": 1, "radish": 1}),
    "c9": ("data/real2sim/droid/ep_c9", {"bowl": 1, "cup": 1, "mallet": 1}),
}
# nouns accepted as the same kind (for scoring only)
SYN = {"block": {"block", "cube", "toy block", "box", "brick"}, "bowl": {"bowl"}, "cup": {"cup", "mug"},
       "apple": {"apple", "tomato", "fruit"}, "radish": {"radish", "vegetable", "beet"},
       "mallet": {"mallet", "hammer"}, "cylinder": {"cylinder", "block", "prism", "peg"},
       "peg stand": {"peg", "stand", "holder", "stick", "post", "rod"}}
MODELS = ["meta/llama-3.2-11b-vision-instruct", "meta/llama-3.2-90b-vision-instruct",
          "meta/llama-4-maverick-17b-128e-instruct", "microsoft/phi-4-multimodal-instruct"]


def ask(img, model, key, max_tokens=800):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]}]}
    req = urllib.request.Request("https://integrate.api.nvidia.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["choices"][0]["message"].get("content") or "", round(time.time() - t0, 1)


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0))["objects"] if m else None
    except (json.JSONDecodeError, KeyError):
        return None


def score(objs, ref):
    """Per reference kind: was it named (accepted noun), and the count given; plus nouns that match nothing."""
    got = {}
    for o in objs:
        n = str(o.get("noun", "")).strip().lower()
        got.setdefault(n, 0)
        got[n] += int(o.get("count", 1) or 1)
    words_ok = all(len(n.split()) <= 2 and not any(c in n for c in ("red", "green", "blue", "yellow", "orange", "black", "white", "wooden", "plastic")) for n in got)
    rows, used = [], set()
    for kind, cnt in ref.items():
        hits = [n for n in got if n in SYN.get(kind, {kind}) or kind in n]
        used |= set(hits)
        rows.append(dict(kind=kind, ref_count=cnt, named_as=hits, vlm_count=sum(got[h] for h in hits)))
    extra = {n: c for n, c in got.items() if n not in used}
    named = sum(1 for r in rows if r["named_as"])
    return dict(nouns=got, noun_only=words_ok, kinds_named=f"{named}/{len(ref)}",
                counts_exact=sum(1 for r in rows if r["named_as"] and r["vlm_count"] == r["ref_count"]),
                per_kind=rows, extra_nouns=extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--no_detect", action="store_true")
    a = ap.parse_args()
    key = os.environ.get("NVIDIA_NIM_API_KEY") or sys.exit("set NVIDIA_NIM_API_KEY")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "vlm_nouns.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.update(prompt=PROMPT, status="exploratory, UNVERIFIED; reference lists by eye")
    imgs = {s: read_frame(os.path.join(ep, "22008760.mp4"), 0) for s, (ep, _) in SCENES.items()}
    for model in a.models.split(","):
        for s, (ep, ref) in SCENES.items():
            k = f"{s}|{model}"
            if k in res.get("runs", {}):
                continue
            try:
                text, secs = ask(imgs[s], model, key)
            except urllib.error.HTTPError as e:
                res.setdefault("runs", {})[k] = dict(error=f"HTTP {e.code}")
                print(k, "HTTP", e.code, flush=True)
                if e.code in (404, 403):
                    break  # model not available for this key: skip its other scenes
                continue
            except Exception as e:  # timeouts etc.
                res.setdefault("runs", {})[k] = dict(error=repr(e)[:200])
                print(k, "error", repr(e)[:120], flush=True)
                continue
            objs = parse(text)
            row = dict(scene=s, model=model, seconds=secs, raw=text[:1500], objects=objs)
            if objs is not None:
                row["score"] = score(objs, ref)
            res.setdefault("runs", {})[k] = row
            json.dump(res, open(path, "w"), indent=1)
            print(k, secs, "s", json.dumps(row.get("score", {}).get("nouns")), row.get("score", {}).get("kinds_named"), flush=True)
    if not a.no_detect:  # detector count per returned noun (one call per noun)
        from perceive import detect_multi
        for k, row in res.get("runs", {}).items():
            if not row.get("objects") or "detector" in row:
                continue
            img = imgs[row["scene"]]
            det = {}
            vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
            for n in row["score"]["nouns"]:
                d = detect_multi(img, [n])
                det[n] = len(d)
                for name, b in d.items():
                    x0, y0, x1, y1 = (int(v) for v in b["box"])
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
                    cv2.putText(vis, f"{n} {b['score']:.2f}", (x0, max(y0 - 4, 12)), 0, 0.5, (0, 255, 255), 1)
            row["detector"] = det
            cv2.imwrite(os.path.join(a.out, f"{row['scene']}_{row['model'].split('/')[-1]}_det.jpg"), vis)
            json.dump(res, open(path, "w"), indent=1)
            print(k, "detector", det, flush=True)


if __name__ == "__main__":
    main()
