"""E7 steps (a, c, d): closed VLM questions at telemetry event frames, level-4 bisection, scoring against sim truth.
Reads the renders of experiments/e7_render.py. Approved 10-09 (user via prim); NIM (Nemotron-Omni) approved as the
primary backend, local Qwen3-VL-8B-Thinking (4-bit) as the fallback. No result travels before a PASS entry in
PIPELINE_CHECKS.md.

  .venvs/cosmos/bin/python experiments/e7_vlm.py tasks                               # CPU: questions + truth
  NVIDIA_NIM_API_KEY=... .venvs/cosmos/bin/python experiments/e7_vlm.py ask --backend nim --workers 4
  .venvs/cosmos/bin/python experiments/e7_vlm.py bisect --backend nim
  .venvs/cosmos/bin/python experiments/e7_vlm.py score                               # CPU
  (local: bash experiments/d_step.sh e7_vlm --gpu .venvs/cosmos/bin/python experiments/e7_vlm.py ask --backend local)

Level 1, per event frame (one still: agent | wrist, as the policy saw it):
  grasp_onset    approach   "The gripper is about to close. Which object is it closing on?"   truth: object nearest
                            the grip site (sim); options: every movable BDDL object + "no object"
  onset_plus_10  held       "Which object, if any, is the robot's gripper holding?"           truth: robosuite
                            _check_grasp; "no object" if none
  release        on_plate   "Is <target> on the plate?"  yes/no                                truth: LIBERO's BDDL
                            check_ontop(target, plate)
  final          on_plate + done "Has the robot completed the instruction?" yes/no              truth: R-047 success
Options: object phrases from the BDDL object list and the instruction. Two objects with the same phrase (LIBERO-
Spatial's two black bowls): the target takes the instruction's own description, the other is "the other black
bowl". Option order is fixed (sorted object names, "no object" last); the order control permutes it.
Level 4 (bisection), episodes with no grasp event, plus `--bisect-extra` failures as method checks: two images
(t = 0 and t), "Compared with the first image, has any object been moved, picked up or knocked over in the second
image?" yes/no. Ask at the final frame; if yes, halve [0, T] toward the first yes (~8 calls). Truth: first t at which
any object is displaced > 2 cm from t = 0. Scored: change found/not found, and |step error| <= 16 (one forward).
Controls (built in; PIPELINE_CHECKS):
  repeat     30 random questions asked twice, identical input -> agreement rate (null);
  order      30 object questions with the option order reversed -> same object chosen (invariance);
  known      30 t = 0 frames, "held" -> truth "no object" (the gripper is open at reset; known answer / false alarm);
  invariant  the render's replay fidelity and replayed-vs-logged object positions are carried into the score.
Outputs runs/root_signals/e7/: tasks.jsonl, answers.jsonl (resumable, keyed by task id), bisect.jsonl, score.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import numpy as np

OUT = "runs/root_signals/e7"
RENDER = os.path.join(OUT, "render")
NAMES = {"akita_black_bowl": "black bowl", "glazed_rim_porcelain_ramekin": "ramekin", "cookies": "cookie box",
         "plate": "plate", "new_salad_dressing": "salad dressing bottle", "chocolate_pudding": "chocolate pudding box"}
NONE = "no object"
MOVE_M = 0.02
APPROACH_MAX_M = 0.03      # approach truth threshold (fixed 10-09, before scoring)
AMBIG_M = 0.01             # second-nearest within 1 cm -> ambiguous (fixed 10-09, before scoring)


def base_phrase(o):
    b = re.sub(r"_\d+$", "", o)
    return NAMES.get(b, b.replace("_", " "))


def target_phrase(instr):
    m = re.match(r"pick up (the .+?) and place", instr)
    return m.group(1) if m else None


def phrases(objs, target, instr):
    """object -> option text. Duplicated base names: the target gets the instruction's description."""
    base = {o: base_phrase(o) for o in objs}
    out = {}
    for o in objs:
        dup = sum(1 for x in objs if base[x] == base[o]) > 1
        if dup:
            out[o] = (target_phrase(instr) or f"the {base[o]}") if o == target else f"the other {base[o]}"
        else:
            out[o] = f"the {base[o]}"
    if len(set(out.values())) != len(out):
        raise RuntimeError(f"ambiguous option phrases: {out}")
    return out


_R047 = None


def r047_instruction(line):
    """The instruction R-047's policy actually received (rollouts.jsonl row["instruction"], the BASE instruction;
    fingerprint instruction_source base_scene_FORCED_unperturbed_control). events.json holds the LIBERO-Plus
    variant's paraphrase (replay() builds its env without libero_plus_base_instruction), which must not reach the
    VLM (my prim, 10-09)."""
    global _R047
    if _R047 is None:
        _R047 = open("runs/r047/rollouts.jsonl").readlines()
    return json.loads(_R047[line])["instruction"]


def load_ep(ep):
    d = os.path.join(RENDER, ep)
    info = json.load(open(os.path.join(d, "events.json")))
    info["instruction_variant_paraphrase"] = info.get("instruction")
    info["instruction"] = r047_instruction(info["line"])
    truth = [json.loads(l) for l in open(os.path.join(d, "truth.jsonl"))]
    return info, truth


def nearest(tr):
    e = np.array(tr["eef"])
    return min(tr["pos"], key=lambda o: np.linalg.norm(np.array(tr["pos"][o]) - e))


def tasks(a):
    eps = sorted(x for x in os.listdir(RENDER) if os.path.exists(os.path.join(RENDER, x, "events.json")))
    rng = np.random.default_rng(9)
    T = []
    for ep in eps:
        info, truth = load_ep(ep)
        objs, tgt, instr = info["objects"], info["target"], info["instruction"]
        if target_phrase(instr) is None:
            raise RuntimeError(f"{ep}: instruction {instr!r} has no 'pick up <target> and place' phrase")
        ph = phrases(objs, tgt, instr)
        obj_opts = [ph[o] for o in sorted(objs)] + [NONE]
        tp = ph[tgt]
        for ev in info["events"]:
            t = ev["t_frame"]
            tr = truth[t]
            img = os.path.join(RENDER, ep, f"ev_{t:03d}.png")
            base = {"ep": ep, "role": info["role"], "axis": info["axis"], "event": ev["event"], "t": t, "images": [img],
                    "instruction": instr}
            if ev["event"] == "grasp_onset":
                # truth: the object with the smallest fingerpad-to-surface distance (e7_truth_contacts.py), <= 3 cm,
                # else "no object". The body-centre rule (nearest()) mislabelled rim grasps of bowls (L014 t=31).
                tp_ = os.path.join(RENDER, ep, "truth_events.json")
                if not os.path.exists(tp_):
                    raise SystemExit(f"{ep}: no truth_events.json; run e7_truth_contacts.py first")
                te = json.load(open(tp_))["steps"][str(t)]
                order = sorted(te, key=lambda o: te[o]["min_pad_dist_m"])
                d1, d2 = te[order[0]]["min_pad_dist_m"], te[order[1]]["min_pad_dist_m"]
                # rules fixed 10-09 before any VLM answer is scored (my prim): nearest fingerpad surface distance
                # <= APPROACH_MAX_M names the object, else "no object"; second nearest within AMBIG_M of the nearest
                # -> "ambiguous", counted but left out of accuracy
                tru = [ph[order[0]]] if d1 <= APPROACH_MAX_M else [NONE]
                amb = (d2 - d1) <= AMBIG_M and d1 <= APPROACH_MAX_M
                T.append({**base, "qtype": "approach", "question": "The gripper is about to close. Which object is it closing on?",
                          "options": obj_opts, "truth": tru, "truth_ambiguous": bool(amb),
                          "truth_min_pad_dist_m": d1, "truth_second_pad_dist_m": d2,
                          "truth_body_centre_rule_deprecated": ph[nearest(tr)]})
            elif ev["event"] == "onset_plus_10":
                T.append({**base, "qtype": "held", "question": "Which object, if any, is the robot's gripper holding?",
                          "options": obj_opts, "truth": [ph[o] for o in tr["held"]] or [NONE]})
            elif ev["event"] in ("release", "final"):
                T.append({**base, "qtype": "on_plate", "question": f"Is {tp} on the plate?", "options": ["yes", "no"],
                          "truth": ["yes" if tr["target_on_plate"] else "no"]})
            if ev["event"] == "final":
                T.append({**base, "qtype": "done", "question": "Has the robot completed the instruction?",
                          "options": ["yes", "no"], "truth": ["yes" if info["success"] else "no"]})
    for i, x in enumerate(T):
        x["id"] = f"{x['ep']}|{x['event']}|{x['t']}|{x['qtype']}"
        assert x["instruction"] == r047_instruction(int(x["ep"][1:])), x["id"]
    assert len({x["id"] for x in T}) == len(T), "duplicate task ids"
    # controls
    C = []
    for x in rng.choice(T, min(30, len(T)), replace=False):
        C.append({**x, "id": x["id"] + "|ctl_repeat", "control": "repeat", "of": x["id"]})
    objq = [x for x in T if x["qtype"] in ("approach", "held")]
    for x in rng.choice(objq, min(30, len(objq)), replace=False):
        C.append({**x, "id": x["id"] + "|ctl_order", "control": "order", "of": x["id"], "options": list(reversed(x["options"]))})
    for ep in rng.choice(eps, min(30, len(eps)), replace=False):
        info, truth = load_ep(ep)
        d = os.path.join(RENDER, ep, "t000.png")
        if not os.path.exists(d):
            import av
            with av.open(os.path.join(RENDER, ep, "frames.mp4")) as c:
                f0 = next(c.decode(video=0)).to_ndarray(format="rgb24")
            from PIL import Image
            Image.fromarray(f0).save(d)
        ph = phrases(info["objects"], info["target"], info["instruction"])
        C.append({"ep": ep, "role": info["role"], "axis": info["axis"], "event": "reset", "t": 0, "images": [d],
                  "instruction": info["instruction"], "qtype": "held", "control": "known_t0",
                  "question": "Which object, if any, is the robot's gripper holding?",
                  "options": [ph[o] for o in sorted(info["objects"])] + [NONE],
                  "truth": [ph[o] for o in truth[0]["held"]] or [NONE], "id": f"{ep}|reset|0|held|ctl_known"})
    with open(os.path.join(OUT, "tasks.jsonl"), "w") as fo:
        for x in T + C:
            fo.write(json.dumps(x) + "\n")
    ap_ = [x for x in T if x["qtype"] == "approach"]
    if ap_:
        dd = np.array([x["truth_min_pad_dist_m"] for x in ap_])
        l14 = {x["t"]: (x["truth"], round(x["truth_min_pad_dist_m"], 4), x["truth_ambiguous"]) for x in ap_ if x["ep"] == "L014"}
        rep = {"n_approach": len(ap_), "n_ambiguous": int(sum(x["truth_ambiguous"] for x in ap_)),
               "n_no_object": int(sum(x["truth"] == [NONE] for x in ap_)),
               "n_differs_from_body_centre_rule": int(sum(x["truth"][0] != x["truth_body_centre_rule_deprecated"] for x in ap_)),
               "nearest_pad_dist_m_quantiles": {q: float(np.quantile(dd, q / 100)) for q in (0, 10, 25, 50, 75, 90, 100)},
               "hist_cm_0_1_2_3_5_10_25": np.histogram(dd * 100, bins=[0, 1, 2, 3, 5, 10, 25.01])[0].tolist(),
               "thresholds": {"approach_max_m": APPROACH_MAX_M, "ambiguous_m": AMBIG_M},
               "known_answer_L014_t31_target_bowl": l14.get(31), "L014_t80": l14.get(80)}
        json.dump(rep, open(os.path.join(OUT, "approach_truth_report.json"), "w"), indent=1)
        print(json.dumps(rep, indent=1))
    print(f"{len(T)} level-1 tasks over {len(eps)} episodes; {len(C)} control tasks;",
          {q: sum(1 for x in T if x["qtype"] == q) for q in ("approach", "held", "on_plate", "done")})


def make_vlm(a):
    from vlm_label import LocalVLM, NimVLM
    if a.backend == "local":
        return LocalVLM(a.model or "Qwen/Qwen3-VL-8B-Thinking", "4bit", a.scale)
    return NimVLM(a.model or "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")


def ask(a):
    from vlm_label import ask_closed
    T = [json.loads(l) for l in open(os.path.join(OUT, "tasks.jsonl"))]
    for x in T:                                   # every prompt carries the instruction R-047's policy received
        if x["instruction"] != r047_instruction(int(x["ep"][1:])):
            raise SystemExit(f"{x['id']}: prompt instruction differs from R-047's row instruction")
    p = os.path.join(OUT, "answers.jsonl")
    done = {json.loads(l)["id"] for l in open(p)} if os.path.exists(p) else set()
    todo = [x for x in T if x["id"] not in done]
    if a.controls_only:                 # the control tasks and the originals they are compared with
        need = {x["of"] for x in T if x.get("control") in ("repeat", "order")}
        todo = [x for x in todo if "control" in x or x["id"] in need]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(done)} answered, {len(todo)} to ask, backend {a.backend}", flush=True)
    vlm = make_vlm(a)
    lock = threading.Lock()
    t0 = time.time()

    def one(x):
        try:
            r = ask_closed(vlm, x["images"], x["instruction"], x["question"], x["options"])
            err = None
        except Exception as e:                              # recorded, retried on the next resume
            r, err = {}, f"{type(e).__name__}: {e}"[:300]
        row = {"id": x["id"], "backend": a.backend, "model": getattr(vlm, "model_id", None), "error": err,
               "choice": r.get("choice"), "choice_index": r.get("choice_index"), "valid": r.get("valid"), "parse_rule": r.get("parse_rule"),
               "text": r.get("text"), "gen_s": r.get("gen_s"), "usage": r.get("usage"), "output_tokens": r.get("output_tokens")}
        if err is None:
            with lock, open(p, "a") as fo:
                fo.write(json.dumps(row) + "\n")
        return row

    workers = a.workers if a.backend == "nim" else 1
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, x) for x in todo]):
            row = fut.result(); n += 1
            if n % 10 == 0 or row["error"]:
                print(f"[{time.time()-t0:6.0f}s] {n}/{len(todo)} {row['id']} -> {row['choice']} {row['error'] or ''}", flush=True)


def bisect(a):
    from vlm_label import ask_closed
    import av
    eps = sorted(x for x in os.listdir(RENDER) if os.path.exists(os.path.join(RENDER, x, "events.json")))
    rng = np.random.default_rng(11)
    sel = []
    for ep in eps:
        info, _ = load_ep(ep)
        if not info["has_grasp_event"]:
            sel.append((ep, "no_grasp_event"))
    fails = [ep for ep in eps if load_ep(ep)[0]["role"] == "failure" and load_ep(ep)[0]["has_grasp_event"]]
    for ep in rng.choice(fails, min(a.bisect_extra, len(fails)), replace=False):
        sel.append((str(ep), "method_check"))
    p = os.path.join(OUT, "bisect.jsonl")
    done = {json.loads(l)["ep"] for l in open(p)} if os.path.exists(p) else set()
    vlm = make_vlm(a)
    q = "Compared with the first image, has any object on the table been moved, picked up or knocked over in the second image?"
    for ep, why in sel:
        if ep in done:
            continue
        info, truth = load_ep(ep)
        with av.open(os.path.join(RENDER, ep, "frames.mp4")) as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        if len(frames) != info["n_frames"]:
            raise RuntimeError(f"{ep}: mp4 has {len(frames)} frames, render recorded {info['n_frames']}")
        p0 = {o: np.array(v) for o, v in truth[0]["pos"].items()}
        moved = [max(np.linalg.norm(np.array(tr["pos"][o]) - p0[o]) for o in p0) > MOVE_M for tr in truth]
        t_true = next((t for t, m in enumerate(moved) if m), None)
        calls = []

        def yes(t):
            r = ask_closed(vlm, [frames[0], frames[t]], info["instruction"], q, ["yes", "no"])
            calls.append({"t": t, "choice": r["choice"], "valid": r["valid"], "gen_s": r.get("gen_s")})
            return r["choice"] == "yes"
        T = len(frames) - 1
        est = None
        if yes(T):
            lo, hi = 0, T
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if yes(mid):
                    hi = mid
                else:
                    lo = mid
            est = hi
        row = {"ep": ep, "why": why, "role": info["role"], "T": T, "t_true_first_move": t_true, "t_est": est,
               "any_move_true": t_true is not None, "any_move_est": est is not None, "n_calls": len(calls), "calls": calls,
               "abs_err": abs(est - t_true) if est is not None and t_true is not None else None}
        with open(p, "a") as fo:
            fo.write(json.dumps(row) + "\n")
        print(ep, why, "true", t_true, "est", est, "calls", len(calls), flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def score(a):
    T = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(OUT, "tasks.jsonl"))}
    A = {}
    for l in open(os.path.join(OUT, "answers.jsonl")):
        r = json.loads(l)
        A[r["id"]] = r                                   # last answer wins on a resumed duplicate
    summ_ = json.load(open(os.path.join(OUT, "render_summary.json")))
    drop = {ep for ep, v in summ_.items() if isinstance(v, dict) and (v.get("max_dev_m") or 0) >= 0.01}     # replay drifted >= 1 cm: excluded
    rows = []
    for i, x in T.items():
        r = A.get(i)
        if r is None:                                    # never answered (e.g. NIM 503 after all retries): its own row
            if x.get("truth_ambiguous") or x["ep"] in drop:
                continue
            rows.append({**x, "choice": None, "valid": False, "correct": False, "no_answer": True, "parse_rule": None})
            continue
        if "parse_rule" not in r:                        # answers written before parse_rule existed: re-derive
            from vlm_label import parse_choice_rule
            r["parse_rule"] = parse_choice_rule(r.get("text") or "", len(x["options"]))[1]
        if x.get("truth_ambiguous"):
            continue                                     # counted in approach_truth_report.json, not scored
        if r["parse_rule"] and not r.get("valid"):        # answer written before the bare-letter rule: re-parse
            from vlm_label import parse_choice_rule
            j, _ = parse_choice_rule(r.get("text") or "", len(x["options"]))
            r = {**r, "valid": True, "choice": x["options"][j]}
        rows.append({**x, "choice": r["choice"], "valid": bool(r["valid"]), "correct": bool(r["valid"] and r["choice"] in x["truth"]),
                     "no_answer": False, "parse_rule": r["parse_rule"]})
    rows = [r for r in rows if r["ep"] not in drop]
    main = [r for r in rows if "control" not in r]
    out = {"what": "E7 level-1 closed questions vs sim truth (exploratory; PIPELINE_CHECKS pending)",
           "n_tasks": sum(1 for x in T.values() if "control" not in x), "n_answered": len(main),
           "backends": sorted({A[r["id"]]["backend"] + ":" + str(A[r["id"]]["model"]) for r in rows})}
    acc = {}
    for q in ("approach", "held", "on_plate", "done"):
        for role in ("failure", "success_calibration"):
            s = [r for r in main if r["qtype"] == q and r["role"] == role]
            k = sum(r["correct"] for r in s)
            acc[f"{q}|{role}"] = {"n": len(s), "accuracy": round(k / len(s), 3) if s else None, "ci95": wilson(k, len(s)),
                                  "invalid": sum(not r["valid"] and not r["no_answer"] for r in s),
                                  "no_answer": sum(r["no_answer"] for r in s),
                                  "rescued_by_bare_letter": sum(r["parse_rule"] == "bare_letter" for r in s),
                                  "accuracy_strict_parse": round(sum(r["correct"] and r["parse_rule"] == "strict" for r in s) / len(s), 3) if s else None,
                                  "note": "accuracy counts invalid and no-answer as wrong (denominator n includes them)"}
    out["accuracy"] = acc
    out["excluded_replay_drift_ge_1cm"] = sorted(drop)
    # false alarms on successes: the VLM reports a problem the sim says isn't there
    fs = [r for r in main if r["role"] == "success_calibration" and r["event"] == "final"]
    fa = [r for r in fs if r["valid"] and r["choice"] == "no"]
    out["success_final_false_alarm"] = {"n": len(fs), "n_said_no": len(fa), "rate": round(len(fa) / len(fs), 3) if fs else None}
    # wrong-object grasps (UQ8): truth holds a non-target object at onset+10
    wrong = []
    for r in main:
        if r["qtype"] != "held":
            continue
        info, _ = load_ep(r["ep"])
        ph = phrases(info["objects"], info["target"], info["instruction"])
        if any(t not in (ph[info["target"]], NONE) for t in r["truth"]):
            wrong.append(r)
    out["wrong_object_held"] = {"n_truth": len(wrong), "n_vlm_named_it": sum(r["correct"] for r in wrong)}
    # confusion for the held question
    conf = {}
    for r in main:
        if r["qtype"] == "held":
            k = f"{'|'.join(r['truth'])} -> {r['choice']}"
            conf[k] = conf.get(k, 0) + 1
    out["held_confusion"] = dict(sorted(conf.items(), key=lambda kv: -kv[1]))
    ctl = {}
    for kind in ("repeat", "order"):
        s = [r for r in rows if r.get("control") == kind and r["of"] in A]
        same = sum(1 for r in s if r["choice"] == A[r["of"]]["choice"])
        ctl[kind] = {"n": len(s), "same_choice": same, "rate": round(same / len(s), 3) if s else None}
    s = [r for r in rows if r.get("control") == "known_t0"]
    ctl["known_t0_held_none"] = {"n": len(s), "correct": sum(r["correct"] for r in s)}
    out["controls"] = ctl
    summ = json.load(open(os.path.join(OUT, "render_summary.json")))
    summ = {k: v for k, v in summ.items() if isinstance(v, dict)}
    dev = [v.get("max_dev_m") for v in summ.values() if v.get("max_dev_m") is not None]
    out["render_invariants"] = {"episodes": len(summ), "replay_max_dev_m_max": max(dev) if dev else None,
                                "episodes_dev_ge_1cm": sum(1 for x in dev if x >= 0.01)}
    bp = os.path.join(OUT, "bisect.jsonl")
    if os.path.exists(bp):
        B = [json.loads(l) for l in open(bp)]
        out["bisect"] = {"n": len(B), "change_detection_agree": sum(b["any_move_true"] == b["any_move_est"] for b in B),
                         "within_one_forward": sum(1 for b in B if b["abs_err"] is not None and b["abs_err"] <= 16),
                         "with_true_change": sum(b["any_move_true"] for b in B),
                         "median_calls": float(np.median([b["n_calls"] for b in B])) if B else None}
    json.dump(out, open(os.path.join(OUT, "score.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tasks", "ask", "bisect", "score"])
    ap.add_argument("--backend", choices=["nim", "local"], default="nim")
    ap.add_argument("--model", default=None)
    ap.add_argument("--scale", type=float, default=0.5, help="local only")
    ap.add_argument("--workers", type=int, default=4, help="nim only: parallel requests")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--controls-only", action="store_true", help="ask: only the control tasks and their originals")
    ap.add_argument("--bisect-extra", type=int, default=10, help="failures with grasp events also bisected, as method checks")
    a = ap.parse_args()
    {"tasks": tasks, "ask": ask, "bisect": bisect, "score": score}[a.phase](a)


if __name__ == "__main__":
    main()
