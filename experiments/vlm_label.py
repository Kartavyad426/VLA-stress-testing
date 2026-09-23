"""Ask a video VLM what happened in an episode, blind to the outcome.

    .venvs/cosmos/bin/python experiments/vlm_label.py <video.mp4> \
        --instruction "pick up the black bowl ..." \
        [--backend local|nim] [--model nvidia/Cosmos-Reason2-2B] [--quant 4bit]

Backends:
  local  transformers, in the isolated `.venvs/cosmos` (bitsandbytes lives there
         and nowhere else, so no policy venv changes). `--quant 4bit` is NF4,
         the only way an 8B model fits the 8 GB card.
  nim    NVIDIA's hosted API. Key from $NVIDIA_NIM_API_KEY. `--nim-frames N`
         sends N stills instead, for models without video input.

The prompt never states the outcome or the perturbation: a label that could be
read off the prompt measures the prompt, not the video. The answer is scored
against simulator truth elsewhere, never trusted on its own.

Frame budget: Cosmos-Reason2 was trained at 4 fps (model card), and a 281-step
LIBERO failure is 14 s -> 56 frames. Frames stay at native 360x720 (~125
tokens per frame after merging).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time

PROMPT = """This video shows a robot arm attempting the instruction: "{instruction}".
Left half: a fixed third-person camera. Right half: a camera mounted on the robot's wrist.

Watch the whole video, then answer:
1. Did the robot complete the instruction? (yes / no)
2. Did the gripper ever close on an object? If so, which object?
3. Did the arm move to the object named in the instruction, or somewhere else?
4. In one or two sentences, how did the attempt unfold over time?

End with a JSON object on its own line:
{{"completed": true|false, "grasped_object": "<name or none>", "approached_target": true|false, "summary": "<one sentence>"}}"""

SYSTEM = ("Answer the question in the following format: <think>\nyour reasoning\n"
          "</think>\n\n<answer>\nyour answer\n</answer>.")      # model card, :268


def parse_json(text: str) -> dict | None:
    """Last {...} in the answer. None rather than a guess if there is none."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.S))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def decode(video: str, fps: float, scale: float = 1.0):
    """Frames at `fps`, decoded here with PyAV.

    transformers 5.5 fetches videos with torchcodec, else falls back to
    `torchvision.io.read_video` -- removed from torchvision 0.26, and the PyPI
    torchcodec wheel targets CUDA 13 (this box is 12.8). Decoding ourselves also
    makes the sampled indices explicit, so the timestamps the model is shown are
    the ones recorded here.
    """
    import av
    import numpy as np
    with av.open(video) as c:
        src_fps = float(c.streams.video[0].average_rate)
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    step = src_fps / fps
    idx = [int(round(i * step)) for i in range(max(1, int(len(frames) / step)))]
    idx = [i for i in idx if i < len(frames)]
    out = [frames[i] for i in idx]
    if scale != 1.0:
        # the vision tower attends over every patch of every frame at once;
        # at native 360x720 x 56 frames it OOMs an 8 GB card next to 4-bit
        # 8B weights, so resolution is the knob that buys frames.
        from PIL import Image
        h, w = out[0].shape[:2]
        size = (int(w * scale) // 32 * 32, int(h * scale) // 32 * 32)
        out = [np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR)) for f in out]
    return np.stack(out), idx, src_fps, len(frames)


def run_local(video: str, text: str, model_id: str, quant: str | None,
              fps: float, max_new_tokens: int, scale: float = 1.0) -> dict:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.video_utils import VideoMetadata

    kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kw).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    load_s = time.time() - t0

    v, idx, src_fps, n_src = decode(video, fps, scale)
    md = VideoMetadata(total_num_frames=n_src, fps=src_fps, width=v.shape[2],
                       height=v.shape[1], duration=n_src / src_fps,
                       video_backend="pyav", frames_indices=idx)
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "video"},
                                         {"type": "text", "text": text}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True)
    inputs = proc(text=[prompt], videos=[v], video_metadata=[md],
                  do_sample_frames=False, return_tensors="pt").to(model.device)
    n_in = int(inputs["input_ids"].shape[1])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0, n_in:]
    return {"text": proc.decode(gen, skip_special_tokens=True),
            "input_tokens": n_in, "output_tokens": int(gen.shape[0]),
            "n_frames": len(idx), "frame_hw": list(v.shape[1:3]), "load_s": round(load_s, 1),
            "gen_s": round(time.time() - t0, 1),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def run_nim(video: str, text: str, model_id: str, max_new_tokens: int,
            n_frames: int = 0) -> dict:
    import urllib.request
    key = os.environ.get("NVIDIA_NIM_API_KEY")
    if not key:
        raise SystemExit("set NVIDIA_NIM_API_KEY")
    if n_frames:
        # models without video input (Gemma): N evenly spaced stills, in order,
        # the protocol FailBench used for such models.
        import io
        import imageio.v3 as iio
        v, _, _, _ = decode(video, 20.0)
        pick = [int(i * (len(v) - 1) / (n_frames - 1)) for i in range(n_frames)]
        media = []
        for i in pick:
            buf = io.BytesIO()
            iio.imwrite(buf, v[i], extension=".jpg")
            media.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
        text = f"The {n_frames} images are frames in time order, evenly spaced.\n" + text
    else:
        b64 = base64.b64encode(open(video, "rb").read()).decode()
        media = [{"type": "video_url",
                  "video_url": {"url": "data:video/mp4;base64," + b64}}]
    body = {"model": model_id, "max_tokens": max_new_tokens,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": [
                             {"type": "text", "text": text}, *media]}]}
    req = urllib.request.Request(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    msg = r["choices"][0]["message"]
    return {"text": msg.get("content") or "", "usage": r.get("usage"),
            "gen_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--backend", choices=["local", "nim"], default="local")
    ap.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    ap.add_argument("--quant", choices=["4bit"], default=None)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="local only: resize frames by this factor (OOM knob)")
    ap.add_argument("--nim-frames", type=int, default=0,
                    help="NIM only: send N stills instead of the mp4 (models without video input)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    a = ap.parse_args()

    text = PROMPT.format(instruction=a.instruction)
    if a.backend == "local":
        res = run_local(a.video, text, a.model, a.quant, a.fps, a.max_new_tokens,
                        a.scale)
    else:
        res = run_nim(a.video, text, a.model, a.max_new_tokens, a.nim_frames)
    res.update({"video": a.video, "model": a.model, "backend": a.backend,
                "quant": a.quant, "fps": a.fps, "scale": a.scale, "parsed": parse_json(res["text"])})
    print(json.dumps(res))


if __name__ == "__main__":
    main()
