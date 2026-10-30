"""Generated-plate test, step 2: exterior plates from FLUX.1 Depth [dev] (NF4), conditioned on the empty-scene depth (GPU).

FLUX.1 [dev] is under the Non-Commercial licence: research test only, the plates inherit it.
Weights: transformer + T5 from diffusers/FLUX.1-Depth-dev-nf4 (bitsandbytes NF4), VAE / CLIP / tokenizers / scheduler
from black-forest-labs/FLUX.1-Depth-dev (gated).

Three phases, three processes, to keep host RAM low (prim's limit ~8 GB; machine RAM is shared with other sessions) and
so that the VAE and the transformer are never on the 8 GB GPU together (10-30: both OOMs came from that):
  embed    CLIP + NF4 T5 loaded straight to the GPU, all prompts encoded once -> <run>/prompt_embeds.pt, process exits
  denoise  VAE alone encodes + packs the depth control, then goes to CPU; the NF4 transformer alone denoises each prompt,
           output_type="latent"; each latent is saved at once -> <run>/latent_<name>.pt
  decode   VAE alone (tiled) decodes every latent_<name>.pt -> <run>/plate_<name>.png
No CPU offload (offload would park the 6 GB transformer in host RAM). A watchdog thread aborts the process (exit 3) if
MemAvailable < --min_avail_gib or the process's anonymous RSS (RssAnon) > --max_rss_gib (prim, 10-30: mapped file pages
are reclaimable and covered by MemAvailable); total RSS is logged too. Refuses to start if the GPU has > 500 MiB in use.

    flock /tmp/vla_gpu.lock sh -c 'P=".venvs/flux/bin/python experiments/genplate/gen_flux.py --run runs/genplate/orange";
                                    $P --phase embed && $P --phase denoise && $P --phase decode'

Writes <run>/plate_<name>.png (size of depth_control.png) and <run>/gen_flux_<phase>.json (prompts, seeds, steps,
guidance, seconds, peak VRAM, peak RssAnon and RSS, min MemAvailable, checkpoints before each GPU step). UNVERIFIED.
"""
import argparse, gc, json, os, subprocess, sys, threading, time

PROMPTS = {
    "oak": "a photo of an empty light oak wooden workbench in a robotics lab, plain white wall behind, nothing on the table, soft indoor ceiling light, realistic",
    "white": "a photo of an empty white laminate table in a bright office, light grey wall behind, nothing on the table, diffuse daylight from a window, realistic",
    "walnut": "a photo of an empty dark walnut wooden table in a kitchen, beige tiled wall behind, nothing on the table, warm indoor light, realistic",
}
OFFICIAL = "black-forest-labs/FLUX.1-Depth-dev"
NF4 = "diffusers/FLUX.1-Depth-dev-nf4"
GIB = 2**30


def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024


def rss(field="VmRSS"):
    for line in open("/proc/self/status"):
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024


class Watchdog(threading.Thread):
    def __init__(self, min_avail, max_rss):
        super().__init__(daemon=True)
        self.min_avail, self.max_rss = min_avail, max_rss
        self.low_avail, self.high_rss, self.high_anon = mem_available(), rss(), rss("RssAnon")

    def run(self):
        while True:
            av, rs, an = mem_available(), rss(), rss("RssAnon")
            self.low_avail, self.high_rss, self.high_anon = min(self.low_avail, av), max(self.high_rss, rs), max(self.high_anon, an)
            if av < self.min_avail or an > self.max_rss:
                print(f"WATCHDOG ABORT: MemAvailable {av / GIB:.2f} GiB, RssAnon {an / GIB:.2f} GiB, RSS {rs / GIB:.2f} GiB", flush=True)
                os._exit(3)
            time.sleep(0.25)

    def stats(self):
        return dict(peak_rss_anon_gib=round(self.high_anon / GIB, 2), peak_rss_gib=round(self.high_rss / GIB, 2),
                    min_mem_available_gib=round(self.low_avail / GIB, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with depth_control.png (render_empty.py); plates go here")
    ap.add_argument("--phase", required=True, choices=["embed", "denoise", "decode"])
    ap.add_argument("--names", default=",".join(PROMPTS), help="which prompts, comma-separated")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=10.0, help="model card example value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_avail_gib", type=float, default=6.0)
    ap.add_argument("--max_rss_gib", type=float, default=8.0, help="cap on RssAnon")
    a = ap.parse_args()

    # prim's gate rule: total GPU memory used <= 500 MiB (a resident system process, launch_vst ~86 MiB, is always listed)
    used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()[0])
    if used > 500:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
        sys.exit(f"GPU in use ({used} MiB), not loading FLUX:\n{apps}")
    wd = Watchdog(a.min_avail_gib * GIB, a.max_rss_gib * GIB)
    wd.start()

    import torch
    from PIL import Image
    from diffusers import FluxControlPipeline, FluxTransformer2DModel
    names = a.names.split(",")
    emb_path = os.path.join(a.run, "prompt_embeds.pt")
    t0 = time.time()
    rec = dict(phase=a.phase, model=f"{OFFICIAL} (transformer, T5 from {NF4})", licence="FLUX.1 [dev] Non-Commercial",
               start_mem_available_gib=round(mem_available() / GIB, 2), tag="UNVERIFIED")
    rec["checkpoints"] = []

    def ckpt(label):
        """state before/after each GPU step, printed and written to the json at once (a crash still leaves numbers)"""
        c = dict(label=label, t_s=round(time.time() - t0, 1), rss_anon_gib=round(rss("RssAnon") / GIB, 2), rss_gib=round(rss() / GIB, 2),
                 mem_available_gib=round(mem_available() / GIB, 2),
                 vram_alloc_gib=round(torch.cuda.memory_allocated() / GIB, 2),
                 vram_peak_gib=round(torch.cuda.max_memory_allocated() / GIB, 2), **wd.stats())
        rec["checkpoints"].append(c)
        print("CKPT", json.dumps(c), flush=True)
        json.dump(rec, open(os.path.join(a.run, f"gen_flux_{a.phase}.json"), "w"), indent=1)

    if a.phase == "embed":
        from transformers import T5EncoderModel
        ckpt("before T5 load")
        t5 = T5EncoderModel.from_pretrained(NF4, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, device_map="cuda")
        pipe = FluxControlPipeline.from_pretrained(OFFICIAL, transformer=None, vae=None, text_encoder_2=t5, torch_dtype=torch.bfloat16)
        pipe.text_encoder.to("cuda")
        rec["load_s"] = round(time.time() - t0, 1)
        out = {}
        with torch.no_grad():
            for n in names:
                pe, pooled, _ = pipe.encode_prompt(prompt=PROMPTS[n], prompt_2=None, device="cuda", max_sequence_length=512)
                out[n] = dict(prompt=PROMPTS[n], prompt_embeds=pe.cpu(), pooled_prompt_embeds=pooled.cpu())
        ckpt("encoded")
        torch.save(out, emb_path)
        rec["prompts"] = {n: PROMPTS[n] for n in names}
    elif a.phase == "denoise":
        emb = torch.load(emb_path)
        ctrl = Image.open(os.path.join(a.run, "depth_control.png")).convert("RGB")
        W, H = ctrl.size
        assert W % 16 == 0 and H % 16 == 0, (W, H)
        # 10-30 first run: VAE encode next to the resident transformer OOMed (7.12 of 7.53 GiB). So the VAE and the
        # transformer never run together: (a) VAE alone encodes + packs the control; the pipeline skips its own encode
        # for an already packed (3-dim) control; (b) transformer alone denoises, output_type="latent";
        # (c) decode in its own process (--phase decode; 10-30 second run: the transformer was not freed in-process).
        pipe = FluxControlPipeline.from_pretrained(OFFICIAL, transformer=None, text_encoder=None, text_encoder_2=None,
                                                   tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16)
        ckpt("before vae to gpu")
        pipe.vae.to("cuda")
        ckpt("before vae encode")
        sf, shift = pipe.vae.config.scaling_factor, pipe.vae.config.shift_factor
        with torch.no_grad():
            x = pipe.image_processor.preprocess(ctrl, height=H, width=W).to("cuda", torch.bfloat16)
            lat = (pipe.vae.encode(x).latent_dist.mode() - shift) * sf  # mode, not a sample: deterministic control
            ctrl_packed = pipe._pack_latents(lat, 1, lat.shape[1], lat.shape[2], lat.shape[3])
        rec["vae_encode_peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
        pipe.vae.to("cpu")
        del x, lat
        gc.collect()
        torch.cuda.empty_cache()
        ckpt("after vae encode, vae on cpu; before transformer load")

        transformer = FluxTransformer2DModel.from_pretrained(NF4, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda")
        pipe = FluxControlPipeline.from_pretrained(OFFICIAL, transformer=transformer, vae=pipe.vae, text_encoder=None, text_encoder_2=None,
                                                   tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16)
        rec.update(load_s=round(time.time() - t0, 1), size_wh=[W, H], steps=a.steps, guidance=a.guidance,
                   control_latent="VAE latent_dist.mode()", plates={})
        ckpt("transformer loaded")
        for n in names:
            ckpt(f"before denoise {n}")
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            lat = pipe(prompt_embeds=emb[n]["prompt_embeds"].to("cuda"), pooled_prompt_embeds=emb[n]["pooled_prompt_embeds"].to("cuda"),
                       control_image=ctrl_packed, height=H, width=W, num_inference_steps=a.steps, guidance_scale=a.guidance,
                       generator=torch.Generator("cpu").manual_seed(a.seed), output_type="latent").images.cpu()
            torch.save(dict(latent=lat, height=H, width=W, prompt=emb[n]["prompt"], seed=a.seed), os.path.join(a.run, f"latent_{n}.pt"))
            rec["plates"][n] = dict(prompt=emb[n]["prompt"], seed=a.seed, denoise_s=round(time.time() - t, 1),
                                    peak_vram_gib=round(torch.cuda.max_memory_allocated() / GIB, 2),
                                    peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GIB, 2))
            print(n, rec["plates"][n], flush=True)
            ckpt(f"saved latent_{n}.pt")
    else:  # decode: a fresh process, so the GPU holds only the VAE
        from diffusers import AutoencoderKL
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.pipelines.flux.pipeline_flux_control import FluxControlPipeline as P
        vae = AutoencoderKL.from_pretrained(OFFICIAL, subfolder="vae", torch_dtype=torch.bfloat16)
        scale = 2 ** (len(vae.config.block_out_channels) - 1)
        proc = VaeImageProcessor(vae_scale_factor=scale * 2)  # as FluxControlPipeline (packing doubles it)
        ckpt("before vae to gpu")
        vae.to("cuda")
        vae.enable_tiling()
        rec["plates"] = {}
        with torch.no_grad():
            for n in names:
                z = torch.load(os.path.join(a.run, f"latent_{n}.pt"))
                ckpt(f"before decode {n}")
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                x = P._unpack_latents(z["latent"].to("cuda"), z["height"], z["width"], scale)  # pipeline passes vae_scale_factor (8)
                x = x / vae.config.scaling_factor + vae.config.shift_factor
                proc.postprocess(vae.decode(x.to(torch.bfloat16), return_dict=False)[0], output_type="pil")[0].save(
                    os.path.join(a.run, f"plate_{n}.png"))
                rec["plates"][n] = dict(decode_s=round(time.time() - t, 1), peak_vram_gib=round(torch.cuda.max_memory_allocated() / GIB, 2))
                print(n, rec["plates"][n], flush=True)
                del x
                gc.collect()
                torch.cuda.empty_cache()
        ckpt("decoded all")
    rec["total_s"] = round(time.time() - t0, 1)
    rec["peak_vram_process_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 2)
    rec.update(wd.stats())
    json.dump(rec, open(os.path.join(a.run, f"gen_flux_{a.phase}.json"), "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
