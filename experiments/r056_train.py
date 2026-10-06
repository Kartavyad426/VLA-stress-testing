"""R-056 training: GR00T N1.7 LoRA (arm a) through third_party/lerobot's lerobot_train.

  PYTHONPATH=$PWD/third_party/LIBERO-plus MUJOCO_GL=egl \
  flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r056_train.py \
      --minted-root data/r054_train48 --minted-repo local/r054_train48 [--lora-r 16] [--resume]
  # R-058: --run-prefix r058 --lora-r 4 (alpha defaults to r); output runs/<prefix>_r<r>_train

lerobot_train runs unchanged; this launcher installs what R-055 requires
before calling it (vla_harness/training/groot_lora.py):
  * the policy is built on the CPU, cast to bf16 (adapter fp32), then moved
    to the GPU; model_params_fp32=False;
  * LoRA via the generic PEFT path, targets = every Linear of the DiT and
    vl_self_attention, r=16, alpha=16, dropout 0.1, modules_to_save=[];
  * the trainable-parameter count is printed and asserted < 25 M, and only
    lora_* tensors may be trainable;
  * activation checkpointing on every DiT / vl_self_attention block (the port's
    own flag is never read);
  * data = the minted export mixed with the original libero_spatial_no_noops
    by sampling weight (--shares, default 1:1), vla_harness/training/mix.py;
  * steps count OPTIMIZER steps: lerobot_train's --steps counts micro-batches,
    so --steps = opt_steps x accumulation, --save_freq likewise; every
    checkpoint's pretrained_model/vla_train_meta.json records the OPTIMIZER step
    (directory names are lerobot_train's micro-step count);
  * the LR schedule (5% warmup + cosine over the optimizer-step count) comes from
    vla_harness/training/schedule.py. lerobot_train steps its scheduler once per
    MICRO-batch, so the scheduler holds the LR constant within each accumulation
    window (R-056's first run built it over optimizer steps and cycled its LR).
    Before training, a CPU subprocess drives the same construction through
    lerobot's own accelerator and update_policy and aborts unless every applied LR
    matches the registered formula within 1e-6. The run records the LR of every
    real optimizer update, re-checks it at each checkpoint, and writes
    schedule_verified plus the LR into vla_train_meta.json; r056_eval refuses an
    adapter without schedule_verified. The cached-features loop gets the same;
  * R-056/R-058 train at micro-batch 1 with accumulation 8 (ruling 2026-09-28);
    batch 2 is a memory readout only.

--cached-features DIR switches to a small custom loop that trains the head
from precomputed backbone outputs (feature_cache.py); only if R-055's memory
gate fails. --report writes peak memory (torch and nvidia-smi), the
trainable-parameter summary and, with --eval-loss-episode, a fixed-seed loss
on one minted episode before and after training (the overfit gate).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())

from vla_harness.training import schedule as S  # noqa: E402  (imports nothing heavy at module level)

CKPT = "nvidia/gr00t17-lerobot-libero_spatial-640"
BASE = "nvidia/GR00T-N1.7-3B"
REPLAY = "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"


# --- GPU memory ------------------------------------------------------------------
class SmiPeak:
    """nvidia-smi's used memory, polled; the peak over the run (whole GPU: hold the lock)."""

    def __init__(self, period=0.5):
        self.peak_mib, self.period, self._stop = 0, period, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.peak_mib = max(self.peak_mib, max(int(x) for x in out.split()))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start(); return self

    def __exit__(self, *exc):
        self._stop.set(); self._t.join(timeout=5)


# --- data ------------------------------------------------------------------------
def build_parts(policy_cfg, a, rename_map=None, tolerance_s=1e-4):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    meta = LeRobotDatasetMetadata(a.minted_repo, root=a.minted_root)
    delta = resolve_delta_timestamps(policy_cfg, meta, rename_map or {})
    eps = [int(x) for x in a.episodes.split(",")] if a.episodes else None
    kw = dict(delta_timestamps=delta, video_backend="pyav", return_uint8=True, tolerance_s=tolerance_s)
    parts = [LeRobotDataset(a.minted_repo, root=a.minted_root, episodes=eps, **kw)]
    names = [a.minted_repo]
    if a.shares and not a.episodes:
        parts.append(LeRobotDataset(a.replay, **kw)); names.append(a.replay)
    return parts, names


def shares_of(a, n_parts):
    s = [float(x) for x in a.shares.split(",")] if a.shares else [1.0]
    return s[:n_parts] if n_parts < len(s) else s


# --- a fixed-seed loss (overfit gate, monitoring) -----------------------------------
def fixed_seed_loss(policy, preprocessor, ds, indices, seed=0) -> float:
    """Mean flow-matching loss over `indices`, tau/epsilon/dropout fixed by seeding per frame,
    in eval mode (LoRA and DiT dropout off)."""
    import torch
    from torch.utils.data import default_collate
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    was = policy.training
    policy.eval()
    tot = 0.0
    with torch.no_grad():
        for k, i in enumerate(indices):
            b = _preprocess_dataset_batch(default_collate([ds[i]]), ds.meta.camera_keys, {}, preprocessor)
            torch.manual_seed(seed * 100_003 + k)
            loss, _ = policy.forward(b)
            tot += float(loss)
    policy.train(was)
    return tot / max(1, len(indices))


def episode_indices(ds, ep: int) -> list[int]:
    """Dataset indices of episode `ep` (relative to a LeRobotDataset possibly restricted to episodes)."""
    e = ds.meta.episodes[ep]
    lo, hi = e["dataset_from_index"], e["dataset_to_index"]
    if getattr(ds, "absolute_to_relative_idx", None):
        return [ds.absolute_to_relative_idx[j] for j in range(lo, hi) if j in ds.absolute_to_relative_idx]
    return list(range(lo, hi))


# --- lerobot_train path --------------------------------------------------------------
def lerobot_argv(a, accum) -> list[str]:
    n_micro = (a.max_micro_steps or a.opt_steps * accum)
    args = [
        f"--policy.path={CKPT}", f"--policy.base_model_path={BASE}", "--policy.embodiment_tag=libero_sim",
        "--policy.model_params_fp32=false", "--policy.tune_projector=false", "--policy.push_to_hub=false",
        "--policy.device=cuda",
        f"--dataset.repo_id={a.minted_repo}", f"--dataset.root={a.minted_root}", "--dataset.video_backend=pyav",
        f"--peft.r={a.lora_r}", f"--peft.lora_alpha={a.lora_alpha}", "--peft.full_training_modules=[]",
        f"--batch_size={a.batch_size}", f"--accelerator.gradient_accumulation.steps={accum}",
        f"--steps={n_micro}", f"--save_freq={a.save_every_opt * accum if not a.max_micro_steps else 10**9}",
        f"--save_checkpoint={'false' if a.no_save else 'true'}", "--log_freq=10",
        "--use_policy_training_preset=false",
        "--optimizer.type=adamw", f"--optimizer.lr={a.lr}", "--optimizer.weight_decay=1e-5",
        "--optimizer.grad_clip_norm=1.0",
        "--scheduler.type=diffuser", f"--scheduler.name={a.lr_schedule}",
        f"--scheduler.num_warmup_steps={0 if a.lr_schedule == 'constant' else math.ceil(a.opt_steps * a.warmup_ratio)}",
        f"--output_dir={a.out}", f"--job_name={os.path.basename(a.out)}", f"--seed={a.seed}",
        f"--num_workers={a.num_workers}", "--wandb.enable=false",
    ]
    return args


def run_lerobot(a) -> dict:
    import torch
    import lerobot.scripts.lerobot_train as T
    from vla_harness.training import groot_lora as G
    from vla_harness.training.mix import MixedDataset, weighted_loader

    accum = max(1, a.effective_batch // a.batch_size)
    state = {"policy": None, "pre": None, "parts": None, "optimizer": None, "sched": None, "lr_trace": [], "opt0": 0,
             "guard": a._guard}
    report = {"mode": "lerobot_train", "batch_size": a.batch_size, "accumulation": accum,
              "effective_batch": a.batch_size * accum, "opt_steps": a.opt_steps}

    G.install_peft_defaults(a.lora_r, a.lora_alpha)
    G.install_bf16_make_policy(T)
    report["lora"] = {"r": a.lora_r, "alpha": a.lora_alpha, "dropout": G.LORA_DROPOUT}

    orig_save = T.save_checkpoint

    def save_with_meta(checkpoint_dir, step, *args, **kw):
        orig_save(checkpoint_dir, step, *args, **kw)
        live = lr_record(a, state, step // accum)
        write_meta(os.path.join(str(checkpoint_dir), "pretrained_model"), a, opt_step=step // accum,
                   micro_step=step, accum=accum, extra=live)
        abort_unless_verified(live, checkpoint_dir)

    T.save_checkpoint = save_with_meta

    def make_datasets(cfg):
        parts, names = build_parts(cfg.trainable_config, a, cfg.rename_map, cfg.tolerance_s)
        state["parts"] = parts
        report["parts"] = [{"name": n, "frames": len(p), "episodes": p.num_episodes} for p, n in zip(parts, names)]
        report["shares"] = shares_of(a, len(parts))
        return MixedDataset(parts, names), None

    def make_loaders(cfg, dataset, eval_dataset, step, parallel_dims):
        state["opt0"] = step // accum                     # the first optimizer step this process runs (resume)
        n = (cfg.steps - step) * cfg.batch_size
        return weighted_loader(dataset, shares_of(a, len(dataset.parts)), cfg.batch_size, n, cfg.seed or 0,
                               num_workers=cfg.num_workers, start_step=step), None

    orig_pp = T.make_pre_post_processors

    def make_pp(*args, **kw):
        pre, post = orig_pp(*args, **kw)
        state["pre"] = pre
        from vla_harness.training.feature_cache import augmentation_report
        report["augmentation"] = augmentation_report(pre)
        return pre, post

    def make_opt(cfg, policy):
        state["policy"] = policy
        if not a.no_grad_ckpt:
            report["checkpointed_blocks"] = G.enable_block_checkpointing(policy)
        report["trainable"] = G.assert_trainable(policy, G.max_trainable(a.lora_r), G.PARAMS_PER_RANK * a.lora_r)
        params = [p for p in policy.parameters() if p.requires_grad]
        optimizer = cfg.optimizer.build(params)
        # NOT cfg.scheduler.build: that counts scheduler steps as optimizer steps, and lerobot_train
        # steps it per micro-batch. The construction the startup guard verified (schedule.py).
        assert cfg.accelerator.gradient_accumulation.steps == accum
        sched = scheduler_builder(a)(optimizer, a.lr_schedule, a.opt_steps, a.warmup_ratio, accum)
        S.record_applied_lr(optimizer, state["lr_trace"])
        state["optimizer"], state["sched"] = optimizer, sched
        report["scheduler"] = {"builder": a.scheduler_builder, "opt_steps": a.opt_steps, "accumulation": accum,
                               "warmup_opt_steps": S.warmup_steps(a.opt_steps, a.warmup_ratio, a.lr_schedule),
                               "stepped": "per micro-batch; LR constant within each accumulation window"}
        if a.eval_loss_episode is not None and state["pre"] is not None:
            ds = state["parts"][0]
            idx = episode_indices(ds, a.eval_loss_episode)
            report["eval_loss_indices"] = len(idx)
            report["eval_loss_step0"] = fixed_seed_loss(policy, state["pre"], ds, idx)
            curve["eval"].append({"opt_step": 0, "fixed_seed_loss": report["eval_loss_step0"]})
            print(f"[r056_train] fixed-seed loss, step 0: {report['eval_loss_step0']:.5f}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        return optimizer, sched

    # --eval-every: the fixed-seed loss curve (overfit diagnostic). Every `eval_every` OPTIMIZER steps the
    # fixed-seed loss on --eval-loss-episode is computed; the training loss is averaged per 10 optimizer steps.
    # Seeding inside the eval re-seeds the global RNG, so a run with --eval-every does not draw the same
    # tau/epsilon sequence as one without it (a diagnostic, not a training run).
    curve = {"eval": [], "train": [], "eval_every": a.eval_every, "episode": a.eval_loss_episode,
             "lr_schedule": a.lr_schedule, "lr": a.lr, "opt_steps": a.opt_steps, "lora_r": a.lora_r}
    micro = {"n": 0, "losses": []}
    orig_update = T.update_policy

    def update_with_curve(*args, **kw):
        tracker, out = orig_update(*args, **kw)
        micro["n"] += 1
        if out and "loss" in out:
            micro["losses"].append(float(out["loss"]))
        if micro["n"] % accum == 0:
            opt_step = micro["n"] // accum
            if opt_step % 10 == 0 and micro["losses"]:
                curve["train"].append({"opt_step": opt_step, "loss_mean10": sum(micro["losses"]) / len(micro["losses"])})
                micro["losses"] = []
            if a.eval_every and opt_step % a.eval_every == 0 and state["policy"] is not None:
                ds = state["parts"][0]
                l = fixed_seed_loss(state["policy"], state["pre"], ds, episode_indices(ds, a.eval_loss_episode))
                curve["eval"].append({"opt_step": opt_step, "fixed_seed_loss": l})
                print(f"[r056_train] opt step {opt_step}: fixed-seed loss {l:.5f}", flush=True)
                if a.curve:
                    json.dump(curve, open(a.curve, "w"), indent=1)
        return tracker, out

    if a.eval_every:
        T.update_policy = update_with_curve
    report["curve"] = curve

    T.make_train_eval_datasets = make_datasets
    T.make_dataloaders = make_loaders
    T.make_pre_post_processors = make_pp
    T.make_optimizer_and_scheduler = make_opt

    if a.resume:
        cfgp = os.path.join(a.out, "checkpoints", "last", "pretrained_model", "train_config.json")
        sys.argv = [sys.argv[0], f"--config_path={cfgp}", "--resume=true"]
    else:
        sys.argv = [sys.argv[0]] + lerobot_argv(a, accum)
    report["argv"] = sys.argv[1:]
    with SmiPeak() as smi:
        T.train()
    report["torch_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["torch_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    report["nvidia_smi_peak_gib"] = smi.peak_mib / 1024
    report["lr_live"] = lr_record(a, state, state["opt0"] + len(state["lr_trace"]))
    report["lr_trace"] = {"first_opt_step": state["opt0"], "applied": state["lr_trace"]}
    if a.eval_loss_episode is not None and state["policy"] is not None:
        ds = state["parts"][0]
        report["eval_loss_final"] = fixed_seed_loss(state["policy"], state["pre"], ds, episode_indices(ds, a.eval_loss_episode))
        print(f"[r056_train] fixed-seed loss, final: {report['eval_loss_final']:.5f}", flush=True)
    if a.curve:
        curve["final_fixed_seed_loss"] = report.get("eval_loss_final")
        json.dump(curve, open(a.curve, "w"), indent=1)
    return report


# --- cached-features path (fallback) -------------------------------------------------
def write_meta(model_dir, a, opt_step, micro_step, accum, extra=None):
    os.makedirs(model_dir, exist_ok=True)
    json.dump({"opt_step": int(opt_step), "micro_step": int(micro_step), "accumulation": int(accum),
               "batch_size": a.batch_size, "effective_batch": a.batch_size * accum, "lora_r": a.lora_r,
               "lora_alpha": a.lora_alpha, "run_prefix": a.run_prefix, "opt_steps_total": a.opt_steps,
               "mode": "cached_features" if a.cached_features else "lerobot_train", **(extra or {})},
              open(os.path.join(model_dir, "vla_train_meta.json"), "w"), indent=1)


# --- the LR schedule: startup guard and live record (schedule.py) ------------------------------

def scheduler_builder(a):
    return getattr(S, a.scheduler_builder)


def schedule_guard(a, accum, mode) -> dict:
    """Simulate the run's schedule on the CPU in a fresh process (accelerate's state is a per-process
    singleton) through the same code path, and abort unless it matches the registered formula."""
    sim = (f"S.simulate_lerobot(S.{a.scheduler_builder}, {a.lr!r}, {a.lr_schedule!r}, {a.opt_steps}, {a.warmup_ratio!r}, {accum})"
           if mode == "lerobot_train" else
           f"S.simulate_cached(S.{a.scheduler_builder}, {a.lr!r}, {a.lr_schedule!r}, {a.opt_steps}, {a.warmup_ratio!r})")
    code = ("import json, sys, warnings; warnings.simplefilter('ignore'); sys.path.insert(0, '.');"
            "from vla_harness.training import schedule as S;"
            f"r = S.check({sim}, {a.lr!r}, {a.lr_schedule!r}, {a.opt_steps}, {a.warmup_ratio!r});"
            "print('GUARD ' + json.dumps(r))")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    line = next((l for l in p.stdout.splitlines() if l.startswith("GUARD ")), None)
    if p.returncode or line is None:
        sys.exit(f"[r056_train] schedule guard could not run (rc {p.returncode}):\n{p.stderr[-3000:]}")
    r = json.loads(line[6:])
    r.update(mode=mode, builder=a.scheduler_builder, accumulation=accum)
    print(f"[r056_train] schedule guard ({mode}, {a.scheduler_builder}, accum {accum}): pass={r['pass']} "
          f"max|lr - registered|={r['max_abs_err']:.3e} over {r['n_updates']} updates", flush=True)
    for k, v in r["probes"].items():
        print(f"[r056_train]   opt step {k:>5}: applied {v['applied']:.6e} registered {v['registered']:.6e} "
              f"|err| {v['abs_err']:.1e}", flush=True)
    if not r["pass"]:
        sys.exit(f"[r056_train] ABORT: the LR schedule does not match the registered one "
                 f"(first bad optimizer step {r['first_bad_step']}, max error {r['max_abs_err']:.3e} > {S.TOL}); "
                 f"nothing was trained")
    return r


def lr_record(a, state, opt_step) -> dict:
    """What a checkpoint records about the LR: the applied trace so far re-checked against the formula,
    the LR of the last update and the scheduler's next LR. schedule_verified needs the startup guard AND
    the live trace to pass."""
    tr = state["lr_trace"]
    live = S.check(tr, a.lr, a.lr_schedule, a.opt_steps, a.warmup_ratio, start=state["opt0"],
                   final=opt_step >= a.opt_steps) if tr else {"pass": False, "max_abs_err": None}
    sched = state.get("sched")
    return {"schedule_verified": bool(state.get("guard", {}).get("pass")) and bool(live["pass"]),
            "schedule": {"type": a.lr_schedule, "lr": a.lr, "warmup_ratio": a.warmup_ratio,
                         "warmup_opt_steps": S.warmup_steps(a.opt_steps, a.warmup_ratio, a.lr_schedule),
                         "builder": a.scheduler_builder},
            "schedule_guard_max_abs_err": state.get("guard", {}).get("max_abs_err"),
            "lr_live_max_abs_err": live["max_abs_err"], "lr_live_updates": len(tr), "lr_live_first_opt_step": state["opt0"],
            "lr_applied_last_update": tr[-1] if tr else None,
            "lr_registered_last_update": S.registered_lr(opt_step - 1, a.lr, a.lr_schedule, a.opt_steps, a.warmup_ratio)
            if opt_step >= 1 else None,
            "lr_next": float(sched.get_last_lr()[0]) if sched is not None else None}


def abort_unless_verified(live, where):
    if not live["schedule_verified"]:
        raise RuntimeError(f"[r056_train] ABORT at {where}: applied LR left the registered schedule "
                           f"(max error {live['lr_live_max_abs_err']}); meta written with schedule_verified=false")


def load_training_policy(device="cuda", backbone_device="cuda", r=16, alpha=16):
    """The training-path policy outside lerobot_train: bf16 base, untrained LoRA adapter (fp32).
    Used by the cached loop and by R-055's zero-step conformance gate."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from vla_harness.training import groot_lora as G
    cfg = PreTrainedConfig.from_pretrained(CKPT)
    cfg.pretrained_path = CKPT
    cfg.base_model_path, cfg.embodiment_tag = BASE, "libero_sim"
    cfg.model_params_fp32 = False
    cfg.tune_projector = False
    cfg.device = "cpu"
    pol = get_policy_class(cfg.type).from_pretrained(CKPT, config=cfg)
    pol = pol.to(torch.bfloat16)
    G.install_peft_defaults(r, alpha)
    peft = pol.wrap_with_peft(peft_cli_overrides={"r": r, "lora_alpha": alpha, "full_training_modules": []})
    for n, p in peft.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    inner = peft.base_model.model
    inner.config.device = device
    inner._groot_model.action_head.to(device)
    inner._groot_model.backbone.to(backbone_device)
    return peft, cfg


def training_preprocessor(cfg, device="cuda"):
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=CKPT,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return pre, post


def run_cached(a) -> dict:
    import torch
    from torch.utils.data import default_collate
    from vla_harness.training import groot_lora as G
    from vla_harness.training.feature_cache import (FeatureCache, collate_backbone, precompute,
                                                    strip_vlm_encode)
    from vla_harness.training.mix import MixedDataset, mixture_weights
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    accum = max(1, a.effective_batch // a.batch_size)
    report = {"mode": "cached_features", "batch_size": a.batch_size, "accumulation": accum, "opt_steps": a.opt_steps,
              "exact": a.batch_size == 1}
    peft, cfg = load_training_policy(r=a.lora_r, alpha=a.lora_alpha)
    report["lora"] = {"r": a.lora_r, "alpha": a.lora_alpha}
    pre, _ = training_preprocessor(cfg)
    parts, names = build_parts(cfg, a)
    if not os.path.exists(os.path.join(a.cached_features, "meta.json")) or a.recompute_cache:
        report["cache"] = precompute(peft.base_model.model, pre, parts, names, a.cached_features, CKPT)
    cache = FeatureCache(a.cached_features)
    report["cache_bytes_per_frame"] = cache.meta.get("bytes_per_frame_measured")
    inner = peft.base_model.model
    inner._groot_model.backbone.to("cpu")                # never runs again: free the GPU
    torch.cuda.empty_cache()
    head = inner._groot_model.action_head
    lite = strip_vlm_encode(pre)
    if not a.no_grad_ckpt:
        report["checkpointed_blocks"] = G.enable_block_checkpointing(peft)
    report["trainable"] = G.assert_trainable(peft, G.max_trainable(a.lora_r), G.PARAMS_PER_RANK * a.lora_r)
    params = [p for p in peft.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    n_opt = a.opt_steps
    sched = scheduler_builder(a)(opt, a.lr_schedule, n_opt, a.warmup_ratio, 1)     # stepped once per update
    state = {"lr_trace": [], "opt0": 0, "sched": sched, "guard": a._guard}
    S.record_applied_lr(opt, state["lr_trace"])
    mix = MixedDataset(parts, names)
    w = mixture_weights(mix, shares_of(a, len(parts)))
    start = 0
    ck_root = os.path.join(a.out, "checkpoints")
    last = os.path.join(ck_root, "last")
    if a.resume and os.path.exists(os.path.join(last, "training_state.pt")):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(peft, load_file(os.path.join(last, "pretrained_model", "adapter_model.safetensors")))
        st = torch.load(os.path.join(last, "training_state.pt"))
        opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"]); start = st["opt_step"]
        state["opt0"] = start
    g = torch.Generator().manual_seed(a.seed * 1_000_003 + start)
    total_micro = (a.max_micro_steps or (n_opt - start) * accum)
    draws = torch.multinomial(w, total_micro * a.batch_size, replacement=True, generator=g).tolist()
    peft.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses = []
    with SmiPeak() as smi:
        for m in range(total_micro):
            idx = draws[m * a.batch_size:(m + 1) * a.batch_size]
            samples, feats = [], []
            for i in idx:
                k, j = mix.locate(i)
                samples.append(parts[k][j]); feats.append(cache.get(k, j))
            b = _preprocess_dataset_batch(default_collate(samples), parts[0].meta.camera_keys, {}, lite)
            inputs = inner._filter_groot_inputs(b, include_action=True)
            ai = head.prepare_input(inputs)
            ai = {k: (v.to("cuda", dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda"))
                  if isinstance(v, torch.Tensor) else v for k, v in ai.items()}
            from transformers.feature_extraction_utils import BatchFeature
            bo = collate_backbone(feats, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = head(bo, BatchFeature(data=ai))
            loss = out["loss"] / accum
            loss.backward()
            losses.append(float(out["loss"]))
            if (m + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step = start + (m + 1) // accum
                if step % 10 == 0:
                    print(f"[cached] opt step {step} loss {sum(losses[-accum*10:])/len(losses[-accum*10:]):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
                if not a.no_save and (step % a.save_every_opt == 0 or step == n_opt):
                    d = os.path.join(ck_root, f"{step:06d}")
                    peft.save_pretrained(os.path.join(d, "pretrained_model"))
                    live = lr_record(a, state, step)
                    write_meta(os.path.join(d, "pretrained_model"), a, opt_step=step, micro_step=step * accum, accum=accum,
                               extra=live)
                    abort_unless_verified(live, d)
                    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "opt_step": step},
                               os.path.join(d, "training_state.pt"))
                    if os.path.islink(last) or os.path.exists(last):
                        os.remove(last)
                    os.symlink(f"{step:06d}", last)
    report["torch_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["nvidia_smi_peak_gib"] = smi.peak_mib / 1024
    report["loss_first"], report["loss_last"] = (losses[0] if losses else None), (losses[-1] if losses else None)
    report["lr_live"] = lr_record(a, state, state["opt0"] + len(state["lr_trace"]))
    report["lr_trace"] = {"first_opt_step": state["opt0"], "applied": state["lr_trace"]}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minted-root", required=True)
    ap.add_argument("--minted-repo", required=True)
    ap.add_argument("--replay", default=REPLAY)
    ap.add_argument("--shares", default="0.5,0.5", help="sampling shares minted,replay; '' = minted only")
    ap.add_argument("--episodes", default=None, help="restrict the minted part to these episodes (no replay)")
    ap.add_argument("--out", default=None, help="default runs/<run-prefix>_r<lora-r>_train")
    ap.add_argument("--run-prefix", default="r056")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=None, help="default = r (R-056: 16/16; R-058: alpha = r)")
    ap.add_argument("--opt-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--effective-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"],
                    help="constant: LR fixed, no warmup (R-055 overfit gate as registered: 300 steps at LR 1e-4)")
    ap.add_argument("--save-every-opt", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-micro-steps", type=int, default=None, help="memory probe: stop after N micro-batches")
    ap.add_argument("--eval-loss-episode", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=0, help="fixed-seed loss every N optimizer steps (needs --eval-loss-episode)")
    ap.add_argument("--curve", default=None, help="write the loss curve JSON here")
    ap.add_argument("--cached-features", default=None)
    ap.add_argument("--recompute-cache", action="store_true")
    ap.add_argument("--report", default=None)
    ap.add_argument("--scheduler-builder", default="build_scheduler", choices=["build_scheduler", "legacy_scheduler"],
                    help="legacy_scheduler is R-056's first (cycling) construction; it exists only so the "
                         "startup guard's refusal of it can be demonstrated")
    a = ap.parse_args()
    if a.lora_alpha is None:
        a.lora_alpha = a.lora_r
    if a.out is None:
        a.out = f"runs/{a.run_prefix}_r{a.lora_r}_train"
    if a.effective_batch % a.batch_size:
        sys.exit("effective batch must be a multiple of the micro-batch")
    accum = a.effective_batch // a.batch_size
    a._guard = schedule_guard(a, accum, "cached_features" if a.cached_features else "lerobot_train")
    rep = run_cached(a) if a.cached_features else run_lerobot(a)
    rep["schedule_guard"] = a._guard
    rep["args"] = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    print(json.dumps({k: v for k, v in rep.items() if k not in ("argv", "args", "lr_trace")}, indent=1, default=str))
    if a.report:
        os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
        json.dump(rep, open(a.report, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
