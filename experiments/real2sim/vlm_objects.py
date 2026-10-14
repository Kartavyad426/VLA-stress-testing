"""Real-to-sim: ask a hosted VLM (NVIDIA NIM) to list the objects in a frame, to replace hand-written detector
prompts. Compares the VLM's list with a reference list (the prompts res31 wrote by hand) when given.

Key: NVIDIA_NIM_API_KEY (robotics_agentic/.env, see experiments/vlm_label.py). Writes <out>/vlm_objects_<model>.json.

    set -a; . ~/Documents/Code/robotics_agentic/.env; set +a
    .venvs/real2sim/bin/python experiments/real2sim/vlm_objects.py --episode data/real2sim/droid/ep_c3 --cam 22008760 \
        --out runs/real2sim/ep_c3 --model meta/llama-3.2-90b-vision-instruct
"""
import argparse, base64, io, json, os, re, sys, time, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perceive import read_frame  # noqa: E402

PROMPT = ("List every separate physical object resting on the main table or cart surface in this image. "
          "Exclude the robot arm and gripper, the table itself, and anything on other furniture or in the background. "
          "Count identical-looking objects individually. Answer ONLY with JSON: "
          '{"objects": [{"name": "<short noun phrase>", "color": "<color>", "count": <int>}]}')


def ask(img, model, key, max_tokens=1024):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta/llama-3.2-90b-vision-instruct")
    a = ap.parse_args()
    key = os.environ.get("NVIDIA_NIM_API_KEY") or sys.exit("set NVIDIA_NIM_API_KEY")
    img = read_frame(os.path.join(a.episode, f"{a.cam}.mp4"), a.frame)
    text, secs = ask(img, a.model, key)
    objs = parse(text)
    res = dict(model=a.model, frame=a.frame, seconds=secs, objects=objs, raw=text)
    json.dump(res, open(os.path.join(a.out, f"vlm_objects_{a.model.split('/')[-1]}.json"), "w"), indent=1)
    print(json.dumps(dict(model=a.model, seconds=secs, total=sum(o.get("count", 1) for o in objs) if objs else None,
                          objects=[(o.get("name"), o.get("color"), o.get("count")) for o in objs] if objs else text[:300])))


if __name__ == "__main__":
    main()
