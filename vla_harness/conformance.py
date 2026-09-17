"""Conformance gate -- is this checkpoint correctly wired to this environment?

Run BEFORE a campaign, not after. A policy-environment mismatch does not crash:
it degrades the success rate silently and uniformly, which looks exactly like a
weak policy. The last one cost a 400-episode campaign to find (F9: rendering at
360x360 against a checkpoint declaring 256x256, worth +17 pp on spatial).

    python -m vla_harness.conformance --checkpoint HuggingFaceVLA/smolvla_libero \
        --n-action-steps 10 --obs-size 256 --control-mode relative

THREE VERDICTS, NOT TWO. Every check returns MATCH, MISMATCH or UNDECLARED.

UNDECLARED is not a pass. A checkpoint that does not state its control mode has
not been verified to match `relative`; it has simply not been checked. Folding
UNDECLARED into MATCH is the exact error recorded as O6 in
IMPLEMENTATION_OVERSIGHTS.md -- a detector with no data for a question degrading
into a confident answer instead of abstaining. It has happened four times in
this codebase; this module is built so it cannot happen here.

The comparison logic is stdlib-only, like the rest of the harness core. Only
fetching a remote config touches `huggingface_hub`, and lazily.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

MATCH, MISMATCH, UNDECLARED = "MATCH", "MISMATCH", "UNDECLARED"

# Severity decides whether the gate refuses. A MISMATCH on an ERROR check stops
# the campaign; on a WARN check it is reported and allowed, because some
# departures from the shipped config are deliberate and documented (see
# DOCUMENTED_OVERRIDES).
ERROR, WARN, INFO = "ERROR", "WARN", "INFO"


# --- what the ENVIRONMENT provides ------------------------------------------
# LeRobot's LIBERO env contract, from the official docs
# (huggingface.co/docs/lerobot/main/en/libero, "Policy inputs and outputs").
# Stated here rather than inferred at runtime so the source is auditable.
LIBERO_ENV = {
    "image_keys": ["observation.images.image",      # agentview_image
                   "observation.images.image2"],    # robot0_eye_in_hand_image
    "state_dim": 8,     # eef pos (3) + axis-angle (3) + gripper qpos (2)
    "action_dim": 7,    # 6-D eef delta + 1-D gripper, Box(-1, 1)
    "fps": 20,
    "control_modes": ("relative", "absolute"),
}


# --- departures from a checkpoint's shipped config that are DOCUMENTED ------
# A checkpoint's config.json is the default, not always the evaluation setting.
# LeRobot's own pi0.5 reproduction overrides n_action_steps on the command line
# and reproduces published numbers. Recorded here WITH a source, so a
# recommendation that departs from the shipped value can say why -- and so an
# undocumented departure stays visible as one.
DOCUMENTED_OVERRIDES = {
    "lerobot/pi05_libero_finetuned": {
        "n_action_steps": {
            "value": 10,
            "source": "huggingface.co/docs/lerobot/main/en/libero, "
                      "'Reproducing published results': --policy.n_action_steps=10, "
                      "'matching the original OpenPI implementation' (97.5% vs 96.85%)",
        },
    },
    # MINERVA's model card, "Evaluate the checkpoint": the documented eval
    # overrides the shipped n_action_steps=8 with 1 AND turns on temporal
    # ensembling. The first version of this gate recommended the shipped 8 --
    # the wrong setting, presented as correct. Caught by session vla-7f.
    #
    # Keyed to the HEADLINE checkpoint, not the repo. The repo ships dozens of
    # ablation and intermediate checkpoints; the command is documented for this
    # one only, so extending it repo-wide would assert an eval setting nobody
    # stated. (The MuJoCo pin in PROSE_REQUIREMENTS is different: it is a
    # property of the environment, so it stays repo-wide.)
    "k1000dai/MINERVA/t05_l1_0.54M": {
        "n_action_steps": {
            "value": 1,
            "source": "HF model card 'Evaluate the checkpoint': "
                      "--policy.n_action_steps=1 (shipped config says 8)",
        },
        "obs_size": {
            "value": 360,
            "source": "model card eval command sets no --env.observation_*; the "
                      "MINERVA fork's LiberoEnv defaults to 360x360 "
                      "(src/lerobot/envs/configs.py:333)",
        },
        "temporal_ensemble_coeff": {
            "value": 0.01,
            "source": "HF model card 'Evaluate the checkpoint': "
                      "--policy.temporal_ensemble_coeff=0.01 (shipped config: null)",
        },
    },
    "nvidia/gr00t17-lerobot-libero_spatial-640": {
        "embodiment_tag": {"value": "libero_sim",
                           "source": "huggingface.co/docs/lerobot/main/en/groot"},
    },
    "nvidia/gr00t17-lerobot-libero_object-640": {
        "embodiment_tag": {"value": "libero_sim",
                           "source": "huggingface.co/docs/lerobot/main/en/groot"},
    },
    "nvidia/gr00t17-lerobot-libero_goal-640": {
        "embodiment_tag": {"value": "libero_sim",
                           "source": "huggingface.co/docs/lerobot/main/en/groot"},
    },
    "nvidia/gr00t17-lerobot-libero_10-640": {
        "embodiment_tag": {"value": "libero_sim",
                           "source": "huggingface.co/docs/lerobot/main/en/groot"},
    },
}

# Facts a checkpoint states only in PROSE (a model card), never in config.json.
# They cannot be machine-checked from the checkpoint, so they are held here with
# a source and surfaced as UNDECLARED-in-config. Adding one is a manual act on
# purpose: prose is where silent mismatches hide.
PROSE_REQUIREMENTS = {
    "k1000dai/MINERVA": {
        "mujoco": {"value": "3.3.2",
                   "source": "model card: 'MuJoCo 3.3.2 is pinned due to renderer "
                             "sensitivity'"},
    },
}


@dataclass
class Check:
    name: str
    verdict: str            # MATCH | MISMATCH | UNDECLARED
    severity: str           # ERROR | WARN | INFO
    declared: object = None
    actual: object = None
    note: str = ""

    @property
    def blocks(self) -> bool:
        return self.verdict == MISMATCH and self.severity == ERROR


@dataclass
class Report:
    checkpoint: str
    checks: list[Check] = field(default_factory=list)
    recommended: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(c.blocks for c in self.checks)

    def counts(self) -> dict:
        out = {MATCH: 0, MISMATCH: 0, UNDECLARED: 0}
        for c in self.checks:
            out[c.verdict] += 1
        return out


# --- loading -----------------------------------------------------------------

def split_hub_id(checkpoint: str) -> tuple[str, str | None]:
    """`owner/repo` or `owner/repo/sub/folder` -> (repo_id, subfolder).

    Some repos ship many models side by side. MINERVA's headline checkpoint is
    `k1000dai/MINERVA/t05_l1_0.54M`; there is no config.json at its root at all.
    """
    parts = checkpoint.strip("/").split("/")
    return "/".join(parts[:2]), ("/".join(parts[2:]) or None)


def load_checkpoint_config(checkpoint: str, revision: str | None = None) -> dict:
    """A local directory, a local config.json, or a Hub id (optionally with subfolder)."""
    if os.path.isdir(checkpoint):
        path = os.path.join(checkpoint, "config.json")
    elif os.path.isfile(checkpoint):
        path = checkpoint
    else:
        from huggingface_hub import hf_hub_download          # lazy: venv-only
        repo_id, sub = split_hub_id(checkpoint)
        path = hf_hub_download(repo_id, "config.json", subfolder=sub,
                               revision=revision)
    with open(path) as f:
        return json.load(f)


def _lookup(table: dict, checkpoint: str) -> dict:
    """Exact id first, then the bare repo -- so a requirement stated for a whole
    repo (MINERVA's MuJoCo pin, from its model card) still applies when a caller
    names one checkpoint inside it."""
    if checkpoint in table:
        return table[checkpoint]
    return table.get(split_hub_id(checkpoint)[0], {})


# --- the checks --------------------------------------------------------------

def _is_placeholder(key: str) -> bool:
    """Camera slots the POLICY fabricates, which no environment supplies.

    pi0/pi0.5/SmolVLA-family configs synthesise `observation.images.empty_camera_N`
    in `validate_features()` (e.g. configuration_smolvla.py:124), and the model
    fills any that are missing with `-1` images under `mask=0`
    (modeling_smolvla.py:373) -- present in the tensor, ignored by attention.

    Treating them as env requirements made the first version of this gate
    REFUSE `lerobot/pi05_libero_finetuned`, a checkpoint LeRobot's own docs
    reproduce at 97.5% through the plain libero env. A gate that refuses a
    documented-working config is worse than none: it teaches people to bypass
    it. Their declared shapes are synthetic too (pi0.5 showed a stray 224x224),
    so they are excluded from the resolution check as well.
    """
    return ".empty_camera_" in key


def _hw(shape):
    """(H, W) from an image shape declared as CHW [3,H,W] or HWC [H,W,3].

    SmolVLA / MINERVA declare CHW; GR00T declares HWC. Taking shape[-2:] blindly
    read GR00T's [256,256,3] as (256, 3) and refused the checkpoint.
    """
    shape = list(shape or [])
    if len(shape) < 2:
        return None
    if len(shape) >= 3 and shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4):
        return tuple(shape[-3:-1])
    return tuple(shape[-2:])


# Policy-side camera key aliases, verified in source. A declared key that is not
# in the env is fine if the policy's processor maps an env key onto it.
KEY_ALIASES = {
    "groot": {
        "observation.images.wrist_image": {
            "from": "observation.images.image2",
            "source": "lerobot/policies/groot/processor_groot.py:1579 -- raw N1.7 "
                      "LIBERO checkpoints name the wrist camera `wrist_image`",
        },
    },
}


def _visual_keys(cfg):
    return {k: v for k, v in (cfg.get("input_features") or {}).items()
            if (v or {}).get("type") == "VISUAL" and not _is_placeholder(k)}


def check(checkpoint: str, cfg: dict, run: dict, env: dict = LIBERO_ENV) -> Report:
    """Compare what the checkpoint DECLARES against what the run WILL DO.

    `run` holds the settings the campaign is about to use: obs_size,
    n_action_steps, control_mode, and optionally mujoco / embodiment_tag.
    """
    rep = Report(checkpoint)
    add = rep.checks.append
    visual = _visual_keys(cfg)

    # 1. image keys -- naming is baked into the normalisation stats, so a key
    #    the env does not supply is not a cosmetic difference.
    if not visual:
        add(Check("image_keys", UNDECLARED, ERROR, None, env["image_keys"],
                  "checkpoint declares no VISUAL input_features"))
    else:
        aliases = KEY_ALIASES.get(cfg.get("type"), {})
        missing, via = [], []
        for k in sorted(set(visual) - set(env["image_keys"])):
            al = aliases.get(k)
            if al and al["from"] in env["image_keys"]:
                via.append(f"{al['from']} -> {k} ({al['source']})")
            else:
                missing.append(k)
        note = (f"env does not supply {missing}" if missing else
                ("matched via verified alias: " + "; ".join(via)) if via else "")
        add(Check("image_keys", MISMATCH if missing else MATCH, ERROR,
                  sorted(visual), env["image_keys"], note))

    # 2. image resolution -- F9. The checkpoint pads to a fixed size, so a
    #    different source resolution changes the object's scale in the input
    #    even though nothing errors.
    sizes = {_hw(v.get("shape")) for v in visual.values()} - {None}
    want = run.get("obs_size")
    if not sizes:
        add(Check("obs_resolution", UNDECLARED, ERROR, None, want,
                  "checkpoint declares no image shape"))
    elif want is None:
        add(Check("obs_resolution", UNDECLARED, ERROR, sorted(sizes), None,
                  "run did not state its render resolution -- LeRobot's "
                  "LiberoEnv CONFIG defaults to 360 while its gym CLASS defaults "
                  "to 256, so 'the default' is not one value"))
    else:
        got = (want, want)
        # A policy that resizes internally (MINERVA/tinyflow: image_resize_shape
        # 144x144 then crop) does not consume the declared shape directly, so a
        # render mismatch is a WARN, not a refusal. The first version refused
        # MINERVA's own documented eval command, which renders at its fork's
        # 360x360 default and produced the published 95.75%. F9's refusal stays
        # for policies with no internal resize, where it measured +17 pp.
        own_resize = {k: cfg[k] for k in ("image_resize_shape", "resize_shape",
                                          "image_size") if cfg.get(k)}
        if got in sizes:
            add(Check("obs_resolution", MATCH, ERROR, sorted(sizes), got))
        elif own_resize:
            add(Check("obs_resolution", MISMATCH, WARN, sorted(sizes), got,
                      f"policy resizes internally ({own_resize}), so this is "
                      f"not the F9 defect; still a different resampling path "
                      f"than training -- prefer the documented eval setting"))
        else:
            add(Check("obs_resolution", MISMATCH, ERROR, sorted(sizes), got,
                      f"F9: render {got} vs declared {sorted(sizes)} -- measured "
                      f"at +17 pp on libero_spatial when corrected"))

    # 3. state and action dimensionality
    st = ((cfg.get("input_features") or {}).get("observation.state") or {}).get("shape")
    if st is None:
        add(Check("state_dim", UNDECLARED, ERROR, None, env["state_dim"]))
    else:
        add(Check("state_dim", MATCH if st[-1] == env["state_dim"] else MISMATCH,
                  ERROR, st[-1], env["state_dim"]))
    ac = ((cfg.get("output_features") or {}).get("action") or {}).get("shape")
    if ac is None:
        add(Check("action_dim", UNDECLARED, ERROR, None, env["action_dim"]))
    else:
        add(Check("action_dim", MATCH if ac[-1] == env["action_dim"] else MISMATCH,
                  ERROR, ac[-1], env["action_dim"]))

    # 4. n_action_steps -- the open-loop horizon. WARN, not ERROR: departing
    #    from the shipped value is sometimes right (pi0.5). But the departure
    #    must be DOCUMENTED, or it is a guess wearing a flag.
    shipped = cfg.get("n_action_steps")
    chunk = cfg.get("chunk_size")
    nas = run.get("n_action_steps")
    doc = _lookup(DOCUMENTED_OVERRIDES, checkpoint).get("n_action_steps")
    if nas is None:
        add(Check("n_action_steps", UNDECLARED, WARN, shipped, None,
                  "run did not state n_action_steps"))
    elif shipped is None:
        add(Check("n_action_steps", UNDECLARED, WARN, None, nas,
                  "checkpoint does not declare n_action_steps"))
    elif nas == shipped:
        add(Check("n_action_steps", MATCH, WARN, shipped, nas))
    elif doc and nas == doc["value"]:
        add(Check("n_action_steps", MATCH, WARN, shipped, nas,
                  f"differs from shipped {shipped}, but is a DOCUMENTED "
                  f"override: {doc['source']}"))
    else:
        # Point at the documented value when one exists, not the shipped one:
        # for pi0.5 the shipped 50 is precisely what LeRobot's reproduction
        # overrides, so "use the shipped value" would be the wrong advice.
        use = (f"use the documented {doc['value']}" if doc
               else f"use the shipped {shipped}")
        add(Check("n_action_steps", MISMATCH, WARN, shipped, nas,
                  f"shipped config is {shipped}; {nas} is an UNDOCUMENTED "
                  f"departure. Changes the open-loop horizon and therefore "
                  f"behaviour. Justify it, or {use}."))
    if chunk is not None and nas is not None and nas > chunk:
        add(Check("n_action_steps<=chunk_size", MISMATCH, ERROR, chunk, nas,
                  "cannot execute more actions than one chunk produces"))

    # 5. control mode -- the docs say it MUST match the checkpoint's action
    #    parameterisation, and no config.json we have seen declares it. So this
    #    is UNDECLARED almost everywhere, and saying so is the point.
    declared_cm = cfg.get("control_mode")
    cm = run.get("control_mode")
    if cm is not None and cm not in env["control_modes"]:
        add(Check("control_mode", MISMATCH, ERROR, env["control_modes"], cm,
                  "not a mode LIBERO supports"))
    elif declared_cm is None:
        add(Check("control_mode", UNDECLARED, WARN, None, cm,
                  "checkpoint does not declare its action parameterisation. "
                  "LeRobot docs: 'make sure the mode matches your policy'. "
                  "Unverified, not verified-correct."))
    else:
        add(Check("control_mode", MATCH if declared_cm == cm else MISMATCH,
                  ERROR, declared_cm, cm))

    # 5b. control frequency. Actions are per-step DELTAS, so a non-20 Hz
    #     --env.fps silently changes how far each action moves the arm, with no
    #     error (lerobot#4614; fix PR #4615 only adds a warning). Known leak
    #     sources: lerobot/libero dataset metadata says fps 10, and
    #     lerobot/smolvla_libero's train_config says env.fps 30.
    fps = run.get("fps")
    if fps is None:
        add(Check("fps", UNDECLARED, WARN, env["fps"], None,
                  "run did not state fps; LIBERO delta actions assume 20 Hz"))
    else:
        add(Check("fps", MATCH if fps == env["fps"] else MISMATCH, ERROR,
                  env["fps"], fps, "" if fps == env["fps"] else
                  "delta actions are scaled per step: a different fps changes "
                  "how far every action moves the arm (lerobot#4614)"))

    # 5b'. parallel tasks. lerobot#4327: --env.max_parallel_tasks > 1 drops
    #      success to 0% (85% vs 0%) via a shared-policy threading bug, still
    #      present locally at scripts/lerobot_eval.py:1009-1012, :1054-1058.
    mpt = run.get("max_parallel_tasks")
    if mpt is not None:
        add(Check("max_parallel_tasks", MATCH if mpt == 1 else MISMATCH, ERROR,
                  1, mpt, "" if mpt == 1 else
                  "lerobot#4327: >1 silently collapses success to 0%"))

    # 5c. MuJoCo version bands that affect EVERY LIBERO checkpoint, independent
    #     of any policy's own pin (vla-7f, primary sources):
    #     >= 3.10.0  mj_fullM signature change; robosuite 1.4.0 crashes at env
    #                creation (controllers/base_controller.py:156).
    #     >= 3.4.0   box-box contact distance fix (MuJoCo 883836848a67) makes the
    #                stored libero_spatial task-5 init state settle tilted:
    #                SmolVLA 80->28 on that task (lerobot#4390, LIBERO#141).
    #     >= 3.3.3   rendering/lighting change; mean first-frame pixel gap vs the
    #                datasets 1.9 -> 34.5 on Object (MINERVA README; LIBERO#88).
    #                Unmeasured for pretrained-backbone policies, so WARN only.
    mj = run.get("mujoco")
    if mj is not None:
        try:
            v = tuple(int(x) for x in str(mj).split(".")[:3])
        except ValueError:
            v = None
        if v is None:
            add(Check("mujoco_band", UNDECLARED, WARN, None, mj, "unparseable version"))
        elif v >= (3, 10, 0):
            add(Check("mujoco_band", MISMATCH, ERROR, "<3.10.0", mj,
                      "robosuite 1.4.0 crashes: mj_fullM signature changed"))
        elif v >= (3, 4, 0):
            add(Check("mujoco_band", MISMATCH, ERROR, "<3.4.0", mj,
                      "box-box contact change corrupts the libero_spatial task-5 "
                      "init state for every policy (lerobot#4390)"))
        elif v >= (3, 3, 3):
            add(Check("mujoco_band", MISMATCH, WARN, "<=3.3.2", mj,
                      "past the 3.3.3 rendering change; large pixel shift on "
                      "Object for scratch-CNN policies, unmeasured for others"))
        else:
            add(Check("mujoco_band", MATCH, WARN, "<=3.3.2", mj))

    # 5d. architecture-defining fields. Declared in config.json, silently change
    #     WHICH MODEL this is, and so which published number (if any) applies.
    #     HuggingFaceVLA/smolvla_libero ships num_vlm_layers=0, which keeps all 32
    #     VLM layers (smolvlm_with_expert.py:102) -- not the paper's 16-layer
    #     headline build. Recorded as INFO so it reaches provenance; it is not a
    #     wiring error, it is a fact about what is being evaluated (vla-7f).
    if "num_vlm_layers" in cfg:
        nvl = cfg["num_vlm_layers"]
        add(Check("num_vlm_layers", MATCH, INFO, nvl, None,
                  "0 = FULL VLM depth, not truncated. Compare only against "
                  "published numbers for the same depth." if nvl == 0 else
                  f"VLM truncated to first {nvl} layers"))

    # 6. normalisation -- informational. NOTE for any future live-state check:
    #    compare state against normaliser mean/std PER TASK or PER SCENE, never
    #    globally. LIBERO's global mean z (~0.76) mixes scene heights; object
    #    scenes sit at z~0.26, about -1.3 sigma, so a global z-score would
    #    false-alarm on every object task (vla-7f)., but a missing mapping is worth seeing:
    #    un-normalisation mismatch manufactures apparent model failures.
    nm = cfg.get("normalization_mapping")
    add(Check("normalization_mapping", MATCH if nm else UNDECLARED, INFO, nm, None,
              "" if nm else "no normalisation mapping declared"))

    # 7. documented overrides beyond n_action_steps (e.g. GR00T embodiment_tag)
    for key, o in _lookup(DOCUMENTED_OVERRIDES, checkpoint).items():
        if key in ("n_action_steps",):
            continue
        val = run.get(key)
        add(Check(key, MATCH if val == o["value"] else
                  (UNDECLARED if val is None else MISMATCH), ERROR,
                  o["value"], val, o["source"]))

    # 8. prose-only requirements (MuJoCo pins)
    for key, o in _lookup(PROSE_REQUIREMENTS, checkpoint).items():
        val = run.get(key)
        add(Check(key, MATCH if val == o["value"] else
                  (UNDECLARED if val is None else MISMATCH), ERROR,
                  o["value"], val, o["source"]))

    rep.recommended = recommend(checkpoint, cfg)
    return rep


def recommend(checkpoint: str, cfg: dict) -> dict:
    """Settings derived from the checkpoint, each with where it came from."""
    rec = {}
    sizes = sorted({_hw(v.get("shape")) for v in _visual_keys(cfg).values()} - {None})
    if len(sizes) == 1:
        h, w = sizes[0]
        rec["env.observation_height"] = (h, "config.json input_features")
        rec["env.observation_width"] = (w, "config.json input_features")
    doc = _lookup(DOCUMENTED_OVERRIDES, checkpoint)
    if "n_action_steps" in doc:
        rec["policy.n_action_steps"] = (doc["n_action_steps"]["value"],
                                        "DOCUMENTED override: " + doc["n_action_steps"]["source"])
    elif cfg.get("n_action_steps") is not None:
        # No documented eval. config.json is the checkpoint's DEFAULT, and for
        # the checkpoints we have checked it disagrees with the documented eval
        # whenever one exists (pi0.5: 50 vs 10; MINERVA: 8 vs 1). So with no
        # documentation this is a guess with a provenance, and it is labelled as
        # one rather than presented as the contract.
        rec["policy.n_action_steps"] = (
            cfg["n_action_steps"],
            "config.json shipped DEFAULT -- no documented eval found; "
            "UNVERIFIED as the eval setting (pi0.5 and MINERVA both override "
            "their shipped value)")
    rec["env.control_mode"] = (cfg["control_mode"], "config.json") if cfg.get("control_mode") \
        else ("relative", "LeRobot default -- UNVERIFIED for this checkpoint")
    rec["env.init_states"] = (True, "LeRobot docs: required for comparable runs")
    rec["env.hard_reset"] = (True, "LeRobot docs: 'use hard resets when reproducing'")
    if "obs_size" in doc:                  # documented eval beats config.json
        v, src = doc["obs_size"]["value"], "DOCUMENTED eval: " + doc["obs_size"]["source"]
        rec["env.observation_height"] = (v, src)
        rec["env.observation_width"] = (v, src)
    for key, o in doc.items():
        if key not in ("n_action_steps", "obs_size"):
            rec[f"policy.{key}"] = (o["value"], o["source"])
    for key, o in _lookup(PROSE_REQUIREMENTS, checkpoint).items():
        rec[key] = (o["value"], o["source"])
    return rec


# --- output ------------------------------------------------------------------

def render(rep: Report) -> str:
    L = [f"Conformance: {rep.checkpoint}", ""]
    w = max(len(c.name) for c in rep.checks)
    for c in rep.checks:
        mark = {MATCH: "ok  ", MISMATCH: "FAIL" if c.severity == ERROR else "warn",
                UNDECLARED: " ?? "}[c.verdict]
        line = f"  [{mark}] {c.name:<{w}}  declared={c.declared!s:<24} run={c.actual!s}"
        L.append(line)
        if c.note:
            L.append(f"         {'':<{w}}  -> {c.note}")
    n = rep.counts()
    L += ["", f"  {n[MATCH]} match, {n[MISMATCH]} mismatch, {n[UNDECLARED]} undeclared"]
    if n[UNDECLARED]:
        L.append("  UNDECLARED is not a pass: those settings are unverified, not correct.")
    L += ["", "Recommended settings:"]
    for k, (v, src) in rep.recommended.items():
        L.append(f"  --{k}={v}".ljust(42) + f"  # {src}")
    L += ["", "GATE: PASS" if rep.ok else "GATE: REFUSE -- fix every FAIL above before running"]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--hub-id", default=None,
                    help="identity for override lookup when --checkpoint is a local "
                         "path, e.g. k1000dai/MINERVA/t05_l1_0.54M")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--obs-size", type=int, default=None)
    ap.add_argument("--control-mode", default=None)
    ap.add_argument("--mujoco", default=None, help="MuJoCo version the run will use")
    ap.add_argument("--embodiment-tag", default=None)
    ap.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--max-parallel-tasks", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_checkpoint_config(a.checkpoint, a.revision)
    run = {"n_action_steps": a.n_action_steps, "obs_size": a.obs_size,
           "control_mode": a.control_mode, "mujoco": a.mujoco,
           "embodiment_tag": a.embodiment_tag,
           "temporal_ensemble_coeff": a.temporal_ensemble_coeff,
           "fps": a.fps,
           "max_parallel_tasks": a.max_parallel_tasks}
    rep = check(a.hub_id or a.checkpoint, cfg, run)
    if a.json:
        print(json.dumps({"checkpoint": rep.checkpoint, "ok": rep.ok,
                          "counts": rep.counts(),
                          "checks": [c.__dict__ for c in rep.checks],
                          "recommended": {k: {"value": v, "source": s}
                                          for k, (v, s) in rep.recommended.items()}},
                         indent=2, default=str))
    else:
        print(render(rep))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
