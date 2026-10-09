"""E6 (EXPLORATORY, not pre-registered): FAST action tokens as mining-spec labels.

ROOT_SIGNAL_EXPERIMENTS.html E6, widened by the user on 10-09 (via prim):
can the LIBERO-fitted FAST action tokenizer give a dictionary of action
features, usable as labels from outside the model (actions only)?

FAST (processing_action_tokenizer.py in the HF cache, read 10-09):
  dct(chunk, axis=time, norm="ortho") per action dim over the WHOLE chunk,
  round(coef * scale) - min_token -> one character per (frequency k, dim d),
  flattened k-major (index = k*7 + d), then byte-level BPE over that string.
So a token is a run of quantised frequency coefficients, not a time segment.
Timing therefore comes from SLIDING WINDOWS over the executed action stream:
every window is tokenised on its own, and we track the window's token bag.

Tokenizer: jadechoghari/fast-libero-tokenizer-mean-std (horizon 10, 7 dims,
scale 10, min_token -203). lerobot/pi0fast-libero-v044 names it with ACTION
MEAN_STD normalisation; the stats used here are lerobot/libero's meta/stats.json
(273,465 frames, all four LIBERO suites) [inferred to be the fitting stats; the
tokenizer's metadata says MEAN_STD but doesn't ship the numbers].

Sections (prim's message, 10-09):
  a  tokens enriched in failed vs successful episodes (episode = unit),
     primary on the window every episode has (steps < 80: the length confound),
     stratified by (axis, magnitude) because failures pile up at large offsets;
  b  tokens enriched in windows covering the symptom anchor (grasp onset) vs
     elsewhere, and failing-onset vs successful-onset windows;
  c  tokens by gripper-rule phase (LiberoPhaseSegmenter), plus a usability test:
     does the bag of tokens predict the phase (episode-grouped CV)?
  d  decode: each reported token back to (frequency, dim) footprints and an
     action-space snippet, with an automatic description where one exists;
  e  boundaries: does the change in the window token bag line up with
     gripper-rule phase boundaries, against a shifted-boundary null?
  Every analysis is also run with the gripper dim zeroed ("arm"), because a
  gripper flip changes the gripper's DCT row trivially.

Inputs: runs/r047/rollouts.jsonl (720 rows; joined by LINE INDEX only, never
rollout_id), data/libero_raw toy-task demos (50; gripper-command events only,
no object state), the tokenizer and stats from the HF cache.

  python experiments/root_signals_e6_fast.py            # analysis + checks
  python experiments/root_signals_e6_fast.py --checks-only

Outputs: runs/root_signals/e6/ (summary.json, checks.json, code_state/).
Not a result until PIPELINE_CHECKS has a PASS; until then tag UNVERIFIED.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
from scipy.fft import dct, idct
from scipy.stats import rankdata, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vla_harness.schema import rollout_from_dict  # noqa: E402
from vla_harness.mining.phases_libero import LiberoPhaseSegmenter  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from root_signals_common import join_r047  # noqa: E402

HF = os.path.expanduser("~/.cache/huggingface/hub")
TOK_REPO = "jadechoghari/fast-libero-tokenizer-mean-std"
TOK2_REPO = "lerobot/fast-action-tokenizer"
STATS = glob.glob(f"{HF}/datasets--lerobot--libero/snapshots/*/meta/stats.json")
R047 = os.path.join(ROOT, "runs/r047/rollouts.jsonl")
DEMO = os.path.join(ROOT, "data/libero_raw/libero_spatial/pick_up_the_black_bowl_between_the_plate_"
                    "and_the_ramekin_and_place_it_on_the_plate_demo.hdf5")
OUT = os.path.join(ROOT, "runs/root_signals/e6")
TOY_TRACES = os.path.join(ROOT, "runs/poc_toy/traces.json")

DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
D = 7
MATCHED_T = 80            # forwards 0-4; R-059 A1's common window
LIFT_M, LIFT_WIN = 0.05, 40   # poc_toy_common.grasp_events criterion
MIN_SUPPORT = 10          # a feature must occur in >= this many episodes
N_PERM = 200
RNG = np.random.default_rng(0)
TAG = "EXPLORATORY, UNVERIFIED (no PIPELINE_CHECKS PASS yet)"


# ---------------------------------------------------------------- tokenizer
def load_tokenizer(repo=TOK_REPO):
    # AutoProcessor.from_pretrained fails under transformers 5.5.4 (it tries to
    # convert a slow tokenizer). Build the repo's own UniversalActionProcessor
    # from its own file, its tokenizer.json and its processor_config.json.
    import importlib.util
    from transformers import PreTrainedTokenizerFast
    snap = glob.glob(f"{HF}/models--{repo.replace('/', '--')}/snapshots/*")[0]
    spec = importlib.util.spec_from_file_location("fast_proc", os.path.join(snap, "processing_action_tokenizer.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = json.load(open(os.path.join(snap, "processor_config.json")))
    bpe = PreTrainedTokenizerFast(tokenizer_file=os.path.join(snap, "tokenizer.json"),
                                  clean_up_tokenization_spaces=False)
    proc = mod.UniversalActionProcessor(bpe, scale=cfg["scale"], vocab_size=cfg["vocab_size"],
                                        min_token=cfg["min_token"], action_dim=cfg.get("action_dim"),
                                        time_horizon=cfg.get("time_horizon"))
    return proc, snap


def load_stats():
    s = json.load(open(STATS[0]))["action"]
    return np.array(s["mean"]), np.array(s["std"])


class Fast:
    """FAST encode with the exact steps of UniversalActionProcessor.__call__,
    batched through the BPE tokenizer (verified equal to proc() in checks)."""

    def __init__(self, proc, mean, std):
        self.proc, self.mean, self.std = proc, mean, std
        self.scale, self.min_token = proc.scale, proc.min_token
        self.alpha_max = None

    def coeffs(self, win_norm):            # (N, W, 7) normalised -> quantised ints
        return np.around(dct(win_norm, axis=1, norm="ortho") * self.scale)

    def encode(self, win_norm, offsets=False):
        q = self.coeffs(win_norm)
        flat = q.reshape(len(q), -1) - self.min_token
        clipped = int((flat < 0).sum())
        strs = ["".join(map(chr, np.maximum(r, 0).astype(int))) for r in flat]
        enc = self.proc.bpe_tokenizer(strs, return_offsets_mapping=offsets)
        return enc, q, clipped

    def norm(self, a, arm_only=False):
        z = (a - self.mean) / self.std
        if arm_only:
            z = z.copy(); z[..., 6] = 0.0
        return z


# ---------------------------------------------------------------- data
def load_r047():
    rows, meta = join_r047()
    seg = LiberoPhaseSegmenter()
    eps = []
    for i, (r, m) in enumerate(zip(rows, meta)):
        S = r["steps"]
        A = np.array([s["action"] for s in S if s.get("action")], dtype=np.float64)
        axis, mag = m["axis"], m["magnitude"]
        segs, info = seg(rollout_from_dict(r))
        lab = np.empty(len(S), dtype=object)
        for sg in segs:
            lab[sg.t_start:sg.t_end + 1] = sg.phase
        tgt = info.get("target")
        objs = [s["obs_state"].get("_gt_scene_object_pos") or s["obs_state"]["_gt_object_pos"] for s in S]
        tz = np.array([o[tgt][2] for o in objs]) if tgt else None
        cmd = np.array([bool(s.get("action")) and s["action"][6] > 0.5 for s in S])
        eps.append(dict(idx=i, key=(m["scene"], axis, mag, m["dir"], m["noise_seed"]),
                        scene=m["scene"], axis=axis, mag=mag,
                        success=bool(r["success"]), A=A, T=len(A), phase=lab[:len(A)],
                        segs=[(sg.phase, sg.t_start, sg.t_end) for sg in segs],
                        events=cmd_events(cmd[:len(A)], tz)))
    return eps


def cmd_events(cmd, tz=None):
    """Grasp onsets (open->close command flips) and releases, with the
    poc_toy_common lift criterion where the target height is known."""
    on, off = [], []
    for t in range(1, len(cmd)):
        if cmd[t] and not cmd[t - 1]:
            ev = {"t": t}
            if tz is not None:
                lift = float(tz[t:t + LIFT_WIN + 1].max() - tz[0])
                ev.update(max_lift_m=round(lift, 4), success=lift >= LIFT_M)
            on.append(ev)
        if cmd[t - 1] and not cmd[t]:
            off.append(t)
    return {"onsets": on, "releases": off}


def load_demos():
    import h5py
    eps = []
    with h5py.File(DEMO, "r") as f:
        names = sorted(f["data"], key=lambda s: int(s.split("_")[1]))
        for n in names:
            A = np.array(f["data"][n]["actions"], dtype=np.float64)
            cmd = A[:, 6] > 0
            ev = cmd_events(cmd)
            lab = np.array(["approach"] * len(A), dtype=object)
            first = ev["onsets"][0]["t"] if ev["onsets"] else len(A)
            lab[cmd] = "closed"
            lab[first:][~cmd[first:]] = "released"
            eps.append(dict(idx=n, success=True, A=A, T=len(A), phase=lab, events=ev,
                            segs=_segs(lab)))
    return eps


def _segs(lab):
    out, cur, st = [], lab[0], 0
    for i in range(1, len(lab)):
        if lab[i] != cur:
            out.append((cur, st, i - 1)); cur, st = lab[i], i
    out.append((cur, st, len(lab) - 1))
    return out


# ---------------------------------------------------------------- tokenise
def tokenise(eps, fast, W, s, arm_only):
    """Per episode: window starts and the token ids of each window."""
    allw, own = [], []
    for e in eps:
        starts = list(range(0, e["T"] - W + 1, s))
        if not starts:
            continue
        allw.append(np.stack([e["A"][t:t + W] for t in starts]))
        own.append((e, starts))
    X = fast.norm(np.concatenate(allw), arm_only)
    enc, q, clipped = fast.encode(X)
    ids = enc["input_ids"]
    out, k = {}, 0
    for e, starts in own:
        out[e["idx"]] = {"starts": starts, "ids": ids[k:k + len(starts)]}
        k += len(starts)
    return out, {"windows": len(ids), "clipped_coeffs": clipped,
                 "coeffs": int(q.size), "mean_tokens_per_window": float(np.mean([len(i) for i in ids]))}


def feats(ids_list, bigrams=True):
    """Set of features present in each window: unigrams and adjacent bigrams."""
    out = []
    for ids in ids_list:
        f = {f"u{t}" for t in ids}
        if bigrams:
            f |= {f"b{a}_{b}" for a, b in zip(ids, ids[1:])}
        out.append(f)
    return out


def rate_matrix(per_ep_windows, min_support=MIN_SUPPORT):
    """per_ep_windows: list of lists of feature sets -> (episodes x features)
    fraction of each episode's windows containing the feature."""
    cnt, sup = [], Counter()
    for wins in per_ep_windows:
        c = Counter()
        for fs in wins:
            c.update(fs)
        cnt.append((c, max(len(wins), 1)))
        sup.update(c.keys())
    names = sorted(k for k, v in sup.items() if v >= min_support)
    col = {n: j for j, n in enumerate(names)}
    M = np.zeros((len(cnt), len(names)), np.float32)
    for i, (c, n) in enumerate(cnt):
        for k, v in c.items():
            j = col.get(k)
            if j is not None:
                M[i, j] = v / n
    return M, names


# ---------------------------------------------------------------- stats
def strat_auroc(M, y, strata):
    """AUROC (y=1 high) per column, pooled over strata weighted by n1*n0
    (van Elteren style). Returns (auroc, ranks, groups) so permutations reuse ranks."""
    groups = []
    for g in np.unique(strata):
        m = strata == g
        if y[m].min() == y[m].max():
            continue
        groups.append((m, rankdata(M[m], axis=0)))
    return _auroc_from(groups, y), groups


def _auroc_from(groups, y):
    num = 0.0; den = 0.0
    for m, R in groups:
        yy = y[m]; n1 = yy.sum(); n0 = len(yy) - n1
        a = (yy @ R - n1 * (n1 + 1) / 2) / (n1 * n0)
        num = num + a * n1 * n0; den += n1 * n0
    return num / den


def perm_fwer(groups, y, strata, n=N_PERM):
    """95th percentile of max |AUROC-0.5| over columns under within-stratum label shuffles."""
    mx = []
    for _ in range(n):
        yp = y.copy()
        for g in np.unique(strata):
            m = np.where(strata == g)[0]
            yp[m] = RNG.permutation(y[m])
        mx.append(np.abs(_auroc_from(groups, yp) - 0.5).max())
    return float(np.percentile(mx, 95))


# ---------------------------------------------------------------- sections
def section_a(eps, tok, matched=True):
    sel = [e for e in eps if e["idx"] in tok]
    wins = []
    for e in sel:
        t = tok[e["idx"]]
        lim = min(e["T"], MATCHED_T) if matched else e["T"]
        W = _W(t, e)
        wins.append(feats([ids for st, ids in zip(t["starts"], t["ids"]) if st + W <= lim]))
    keep = [i for i, w in enumerate(wins) if w]
    sel = [sel[i] for i in keep]; wins = [wins[i] for i in keep]
    M, names = rate_matrix(wins)
    y = np.array([0 if e["success"] else 1 for e in sel])
    strata = np.array([f"{e['axis']}|{e['mag']}" for e in sel])
    auc, groups = strat_auroc(M, y, strata)
    thr = perm_fwer(groups, y, strata)
    order = np.argsort(-np.abs(auc - 0.5))[:15]
    by_axis = {}
    for ax in ("joint_radius_rad", "camera_yaw_deg"):
        m = np.array([e["axis"] == ax for e in sel])
        a2, g2 = strat_auroc(M[m], y[m], strata[m])
        by_axis[ax] = {"n_fail": int(y[m].sum()), "n_succ": int((1 - y[m]).sum()),
                       "max_abs_auroc_minus_half": float(np.abs(a2 - 0.5).max()),
                       "fwer95_threshold": perm_fwer(g2, y[m], strata[m])}
    return {"window": f"steps < {MATCHED_T}" if matched else "whole episode (LENGTH-CONFOUNDED)",
            "n_fail": int(y.sum()), "n_succ": int((1 - y).sum()), "n_features": len(names),
            "fwer95_threshold_abs_auroc_minus_half": thr,
            "n_features_passing_fwer": int((np.abs(auc - 0.5) > thr).sum()),
            "top": [{"feature": names[j], "auroc_fail_high": round(float(auc[j]), 3),
                     "passes_fwer": bool(abs(auc[j] - 0.5) > thr),
                     "rate_fail": round(float(M[y == 1, j].mean()), 3),
                     "rate_succ": round(float(M[y == 0, j].mean()), 3)} for j in order],
            "by_axis": by_axis}


def _W(t, e):
    return e["_W"]


def anchor_of(e):
    """Symptom anchor: first failing onset in a failure, first onset in a success."""
    on = e["events"]["onsets"]
    if not on:
        return None
    if e["success"]:
        return on[0]["t"]
    bad = [o for o in on if o.get("success") is False]
    return bad[0]["t"] if bad else None


def section_b(eps, tok):
    rows, contrast_wins, contrast_y, contrast_strata = [], [], [], []
    per_ep = []
    for e in eps:
        if e["idx"] not in tok:
            continue
        ta = anchor_of(e)
        if ta is None:
            continue
        t, W = tok[e["idx"]], e["_W"]
        f = feats(t["ids"])
        at = [fs for st, fs in zip(t["starts"], f) if st <= ta < st + W]
        far = [fs for st, fs in zip(t["starts"], f) if st + W <= ta - W or st >= ta + W]
        if not at or not far:
            continue
        per_ep.append((e, at, far))
        contrast_wins.append(at); contrast_y.append(0 if e["success"] else 1)
        contrast_strata.append(f"{e['axis']}|{e['mag']}")
    out = {}
    # within-episode: anchor windows vs far windows, per group
    for grp, cond in (("successful_first_onset", lambda e: e["success"]),
                      ("failing_first_onset", lambda e: not e["success"])):
        sub = [(e, a, f) for e, a, f in per_ep if cond(e)]
        if len(sub) < 5:
            out[grp] = {"n_episodes": len(sub)}; continue
        Ma, names = rate_matrix([a for _, a, _ in sub] + [f for _, _, f in sub])
        n = len(sub); d = Ma[:n] - Ma[n:]
        mean = d.mean(0); pos = (d > 0).mean(0)
        order = np.argsort(-np.abs(mean))[:10]
        top = []
        for j in order:
            try:
                p = float(wilcoxon(d[:, j]).pvalue)
            except ValueError:
                p = 1.0
            top.append({"feature": names[j], "mean_rate_diff_anchor_minus_far": round(float(mean[j]), 3),
                        "frac_episodes_higher_at_anchor": round(float(pos[j]), 3), "wilcoxon_p": p})
        out[grp] = {"n_episodes": n, "n_features": len(names), "top": top}
    # between episodes: failing-onset windows vs successful-onset windows
    y = np.array(contrast_y); strata = np.array(contrast_strata)
    M, names = rate_matrix(contrast_wins)
    auc, groups = strat_auroc(M, y, strata)
    if groups:
        thr = perm_fwer(groups, y, strata)
        order = np.argsort(-np.abs(auc - 0.5))[:10]
        out["failing_vs_successful_onset"] = {
            "n_fail": int(y.sum()), "n_succ": int((1 - y).sum()), "fwer95_threshold": thr,
            "n_features_passing_fwer": int((np.abs(auc - 0.5) > thr).sum()),
            "top": [{"feature": names[j], "auroc_failing_onset_high": round(float(auc[j]), 3),
                     "passes_fwer": bool(abs(auc[j] - 0.5) > thr)} for j in order]}
    return out


def window_phase(e, st, W):
    lab = Counter(e["phase"][st:st + W])
    return lab.most_common(1)[0][0]


def section_c(eps, tok):
    per_phase = defaultdict(list)
    X, Y, G = [], [], []
    for e in eps:
        if e["idx"] not in tok:
            continue
        t, W = tok[e["idx"]], e["_W"]
        f = feats(t["ids"], bigrams=False)
        ph = [window_phase(e, st, W) for st in t["starts"]]
        for p in set(ph):
            a = [fs for fs, q in zip(f, ph) if q == p]
            b = [fs for fs, q in zip(f, ph) if q != p]
            if a and b:
                per_phase[p].append((a, b))
        for fs, p in zip(t["ids"], ph):
            X.append(fs); Y.append(p); G.append(zlib.crc32(str(e["idx"]).encode()) % 5)
    out = {"enrichment": {}}
    for p, pairs in sorted(per_phase.items()):
        if len(pairs) < 5:
            continue
        Ma, names = rate_matrix([a for a, _ in pairs] + [b for _, b in pairs])
        n = len(pairs); d = Ma[:n] - Ma[n:]
        mean = d.mean(0); order = np.argsort(-mean)[:5]
        out["enrichment"][p] = {"n_episodes": n, "top": [
            {"feature": names[j], "mean_rate_diff_in_minus_out": round(float(mean[j]), 3),
             "frac_episodes_higher_in_phase": round(float((d[:, j] > 0).mean()), 3)} for j in order]}
    out["phase_from_tokens"] = nb_cv(X, Y, np.array(G))
    return out


def nb_cv(X, Y, G):
    """Multinomial naive Bayes on token counts -> window phase; 5 folds grouped by episode."""
    vocab = sorted({t for ids in X for t in ids}); col = {t: j for j, t in enumerate(vocab)}
    C = np.zeros((len(X), len(vocab)), np.float32)
    for i, ids in enumerate(X):
        for t in ids:
            C[i, col[t]] += 1
    cls = sorted(set(Y)); Yi = np.array([cls.index(y) for y in Y])
    pred = np.empty(len(Y), int)
    for k in range(5):
        tr, te = G != k, G == k
        if te.sum() == 0:
            continue
        lp = np.log(np.bincount(Yi[tr], minlength=len(cls)) + 1.0)
        th = np.stack([C[tr][Yi[tr] == c].sum(0) + 1.0 for c in range(len(cls))])
        th = np.log(th / th.sum(1, keepdims=True))
        pred[te] = np.argmax(C[te] @ th.T + lp, 1)
    acc = float((pred == Yi).mean())
    rec = [float((pred[Yi == c] == c).mean()) for c in range(len(cls)) if (Yi == c).any()]
    maj = float(np.bincount(Yi).max() / len(Yi))
    return {"classes": cls, "n_windows": len(Y), "accuracy": round(acc, 3),
            "balanced_accuracy": round(float(np.mean(rec)), 3), "majority_baseline": round(maj, 3),
            "per_class_recall": {c: round(r, 3) for c, r in zip(cls, rec)}}


def change_series(t, W):
    """c(b) = 1 - Jaccard(token set of window ending at b, window starting at b)."""
    starts = t["starts"]; pos = {st: i for i, st in enumerate(starts)}
    bs, cs = [], []
    for st in starts:
        b = st + W
        if b in pos:
            A, B = set(t["ids"][pos[st]]), set(t["ids"][pos[b]])
            bs.append(b); cs.append(1 - len(A & B) / max(len(A | B), 1))
    return np.array(bs), np.array(cs)


def boundaries(e, kind):
    if kind == "phase":
        return [st for _, st, _ in e["segs"][1:]]
    return [o["t"] for o in e["events"]["onsets"]] + list(e["events"]["releases"])


def ep_auroc(bs, cs, bnd, tol):
    if len(bs) < 3 or not bnd:
        return None
    pos = np.array([min(abs(b - x) for x in bnd) <= tol for b in bs])
    if pos.all() or not pos.any():
        return None
    r = rankdata(cs); n1 = pos.sum(); n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def section_e(eps, tok, kind, s):
    ser = {}
    for e in eps:
        if e["idx"] in tok:
            ser[e["idx"]] = (e, *change_series(tok[e["idx"]], e["_W"]))
    obs = [ep_auroc(bs, cs, boundaries(e, kind), s) for e, bs, cs in ser.values()]
    obs = [a for a in obs if a is not None]
    null = []
    for _ in range(N_PERM):
        vals = []
        for e, bs, cs in ser.values():
            bnd = boundaries(e, kind)
            if len(bs) < 3 or not bnd:
                continue
            sh = int(RNG.integers(1, max(e["T"], 2)))
            vals.append(ep_auroc(bs, cs, [(b + sh) % e["T"] for b in bnd], s))
        vals = [v for v in vals if v is not None]
        null.append(np.mean(vals) if vals else np.nan)
    null = np.array(null)
    m = float(np.mean(obs)) if obs else float("nan")
    return {"boundaries": kind, "n_episodes": len(obs), "mean_episode_auroc": round(m, 3),
            "null_mean": round(float(np.nanmean(null)), 3), "null_p95": round(float(np.nanpercentile(null, 95)), 3),
            "p_value": float((np.sum(null >= m) + 1) / (len(null) + 1))}


# ---------------------------------------------------------------- decode
def describe_feature(fname, eps, tok, fast, W, arm_only, max_occ=1000):
    """Footprint of a token (or bigram) in coefficient space, its average
    action-space contribution (env units) and the mean action of windows containing it."""
    ids = [int(x) for x in fname[1:].split("_")]
    occ, contrib, wmean = Counter(), [], []
    allmean = []
    n = 0
    for e in eps:
        if e["idx"] not in tok or n >= max_occ:
            continue
        t = tok[e["idx"]]
        for st, seq in zip(t["starts"], t["ids"]):
            allmean.append(e["A"][st:st + W].mean(0))
            hit = [i for i in range(len(seq) - len(ids) + 1) if seq[i:i + len(ids)] == ids]
            if not hit:
                continue
            n += 1
            win = fast.norm(e["A"][st:st + W][None], arm_only)
            enc, q, _ = fast.encode(win, offsets=True)
            offs = enc["offset_mapping"][0]
            o0, o1 = offs[hit[0]][0], offs[hit[0] + len(ids) - 1][1]
            mask = np.zeros(W * D, bool); mask[o0:o1] = True
            for i in range(o0, o1):
                occ[(i // D, DIMS[i % D])] += 1
            c = q[0].reshape(-1).copy(); c[~mask] = 0
            contrib.append(idct(c.reshape(W, D) / fast.scale, axis=0, norm="ortho") * fast.std)
            wmean.append(e["A"][st:st + W].mean(0))
    if not contrib:
        return {"feature": fname, "occurrences": 0}
    cb = np.mean(contrib, 0)                      # (W, 7) env-unit contribution
    energy = (cb ** 2).sum(0); tot = energy.sum()
    dshare = {DIMS[d]: round(float(energy[d] / tot), 3) for d in np.argsort(-energy)[:3]} if tot > 0 else {}
    ks = Counter(); [ks.update({k: v}) for (k, _), v in occ.items()]
    kdom = ks.most_common(1)[0][0]
    shift = np.mean(wmean, 0) - np.mean(allmean, 0)
    text, describable = _text(cb, energy, tot, kdom, shift)
    return {"feature": fname, "occurrences": n, "decoded_chars": [ord(ch) + fast.min_token for ch in
                                                                     fast.proc.bpe_tokenizer.decode(ids)],
            "footprint_top": [f"k{k}:{d}" for (k, d), _ in occ.most_common(6)],
            "contribution_energy_share": dshare, "dominant_frequency_k": int(kdom),
            "mean_action_shift_vs_all_windows": {DIMS[d]: round(float(shift[d]), 3) for d in range(D)},
            "describable": describable, "description": text}


def _text(cb, energy, tot, k, shift):
    if tot < 1e-10:
        return ("all footprint coefficients quantise to 0: marks the ABSENCE of motion at "
                f"frequency k~{k} (smoothness), not a motion", True)
    top = np.argsort(-energy)[:2]
    share = energy[top].sum() / tot
    if share < 0.7:
        return (f"spread over dims {[DIMS[d] for d in top]}+ ({share:.0%} in top 2): not describable", False)
    parts = []
    for d in top:
        if energy[d] / tot < 0.15:
            continue
        mean, ramp = cb[:, d].mean(), cb[-1, d] - cb[0, d]
        if k == 0:
            what = {0: "x", 1: "y", 2: "z"}.get(d)
            if d == 6:
                parts.append("gripper held " + ("closed" if mean > 0 else "open"))
            elif what:
                parts.append(f"steady {'+' if mean > 0 else '-'}{what} translation ({mean:+.3f}/step)")
            else:
                parts.append(f"steady {DIMS[d]} rotation ({mean:+.3f}/step)")
        elif k == 1:
            parts.append(f"{DIMS[d]} ramp ({ramp:+.3f} over the window)"
                         + (": gripper switching " + ("to close" if ramp > 0 else "to open") if d == 6 else ""))
        else:
            parts.append(f"{DIMS[d]} oscillation/jerk at k={k} (amp {np.abs(cb[:, d]).max():.3f})")
    return "; ".join(parts) or "low-energy", True


# ---------------------------------------------------------------- provenance
def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_state(out):
    d = os.path.join(out, "code_state"); os.makedirs(d, exist_ok=True)
    g = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout
    open(os.path.join(d, "HEAD"), "w").write(g("rev-parse", "HEAD"))
    open(os.path.join(d, "status.txt"), "w").write(g("status", "--porcelain"))
    open(os.path.join(d, "uncommitted.patch"), "w").write(g("diff", "HEAD"))
    import shutil
    shutil.copy(os.path.abspath(__file__), d)


# ---------------------------------------------------------------- checks
def run_checks(eps, demos, fast, proc):
    ck = {}
    # 1 known answer
    n = len(eps); ns = sum(e["success"] for e in eps)
    toy = json.load(open(TOY_TRACES))
    e14 = eps[14]
    on14 = [o["t"] for o in e14["events"]["onsets"]]
    ref_on = [ev["t"] for ev in eval(toy["14"]["events"])] if isinstance(toy["14"]["events"], str) \
        else [ev["t"] for ev in toy["14"]["events"]]
    ref_ph = [list(x) for x in toy["phases_failure"]]
    ck["1_known_answer"] = {
        "r047_rows": n, "r047_successes": ns, "expect": [720, 544], "pass_counts": [n, ns] == [720, 544],
        "row14_onsets": on14, "toy_onsets": ref_on, "pass_onsets": on14 == ref_on,
        "row14_first_failing_onset": anchor_of(e14), "pass_root_t31": anchor_of(e14) == 31,
        "row14_phases_equal_toy": [list(x) for x in e14["segs"]] == ref_ph}
    # 4 invariants: our batched encode == proc(); round trip within quantisation
    w = fast.norm(np.stack([eps[0]["A"][0:10], eps[14]["A"][30:40]]))
    mine = fast.encode(w)[0]["input_ids"]
    ref = proc(w)
    dec = proc.decode(ref, time_horizon=10, action_dim=7)
    keys = [e["key"] for e in eps]
    bound = 0.5 / fast.scale * np.sqrt(10)        # per-sample bound from per-coef rounding
    ck["4_invariants"] = {
        "batched_encode_equals_proc": [list(a) for a in mine] == [list(a) for a in ref],
        "roundtrip_max_abs_norm_units": float(np.abs(dec - w).max()), "roundtrip_bound": float(bound),
        "pass_roundtrip": bool(np.abs(dec - w).max() <= bound + 1e-9),
        "unique_join_keys": len(set(keys)) == len(keys), "n_keys": len(keys),
        "demos": len(demos), "actions_shape_ok": all(e["A"].shape[1] == 7 for e in eps + demos)}
    # 2 null: identical input twice; success-vs-success random split
    enc = fast.encode(fast.norm(np.stack([e["A"][0:10] for e in eps[:50]])), offsets=True)[0]
    ok = all(o[0][0] == 0 and o[-1][1] == 70 and all(a[1] >= b[0] for a, b in zip(o, o[1:]))
             for o in enc["offset_mapping"])
    ck["4_invariants"]["offsets_cover_all_70_coeffs_contiguously"] = bool(ok)
    ck["2_null"] = {"same_input_same_tokens": fast.encode(w)[0]["input_ids"] == mine}
    return ck


def planted(eps, fast, W, s):
    """Two plants into half of the start-pose successes (the other half is clean):
      osc : a dy oscillation (period 4 steps, amp 0.3 env units) at steps [40, 56);
      step: a constant +0.4 dz offset at steps [40, 70) (a step on and a step off).
    Pass for enrichment: a feature passing FWER whose footprint includes the
    planted dim. Pass for timing: the bag-change score peaks at the plant's edges
    (episode AUROC above the shifted-boundary null)."""
    succ = [e for e in eps if e["success"] and e["axis"] == "joint_radius_rad" and e["T"] >= 90]
    idx = RNG.permutation(len(succ)); half = set(idx[: len(succ) // 2].tolist())
    out = {}
    for kind, (lo, hi, dim) in {"osc": (40, 56, 1), "step": (40, 70, 2)}.items():
        eps2 = []
        for i, e in enumerate(succ):
            A = e["A"].copy()
            if i in half:
                t = np.arange(lo, hi)
                A[lo:hi, dim] += (0.3 * np.sin(2 * np.pi * (t - lo) / 4 + np.pi / 4)) if kind == "osc" else 0.4
            eps2.append(dict(e, A=A, idx=("p", kind, e["idx"]), success=i not in half, _W=W,
                             segs=[("x", 0, lo - 1), ("planted", lo, hi - 1), ("x", hi, e["T"] - 1)]))
        tok, _ = tokenise(eps2, fast, W, s, False)
        wins = [feats([ids for st, ids in zip(tok[e["idx"]]["starts"], tok[e["idx"]]["ids"])
                       if st < hi and st + W > lo]) for e in eps2]
        M, names = rate_matrix(wins)
        y = np.array([0 if e["success"] else 1 for e in eps2]); strata = np.zeros(len(y))
        auc, groups = strat_auroc(M, y, strata)
        thr = perm_fwer(groups, y, strata, 100)
        pl = [e for e in eps2 if not e["success"]]
        top = [j for j in np.argsort(-np.abs(auc - 0.5))[:10] if abs(auc[j] - 0.5) > thr]
        descs = [describe_feature(names[j], pl, tok, fast, W, False, 200) for j in top]
        hit = [d["feature"] for d in descs if any(f.endswith(":" + DIMS[dim]) for f in d.get("footprint_top", []))]
        bnd = section_e(pl, tok, "phase", s)
        out[kind] = {"planted_dim": DIMS[dim], "steps": [lo, hi], "n_planted": len(pl),
                     "n_clean": len(eps2) - len(pl), "fwer95": round(thr, 3),
                     "n_features_passing_fwer": int((np.abs(auc - 0.5) > thr).sum()),
                     "top": [{"feature": d["feature"], "auroc": round(float(auc[j]), 3),
                              "footprint_top": d.get("footprint_top"), "description": d.get("description")}
                             for j, d in zip(top, descs)],
                     "features_touching_planted_dim": hit, "pass_enrichment": bool(hit),
                     "boundary_at_edges": bnd,
                     "pass_timing": bool(bnd["mean_episode_auroc"] > bnd["null_p95"])}
    return out


def null_split(eps, tok):
    """Successes only, random fail/success labels: no feature should pass FWER."""
    succ = [e for e in eps if e["success"] and e["idx"] in tok]
    wins = [feats([ids for st, ids in zip(tok[e["idx"]]["starts"], tok[e["idx"]]["ids"])
                   if st + e["_W"] <= MATCHED_T]) for e in succ]
    M, names = rate_matrix(wins)
    y = np.zeros(len(succ), int); y[RNG.permutation(len(succ))[: len(succ) // 3]] = 1
    strata = np.array([f"{e['axis']}|{e['mag']}" for e in succ])
    auc, groups = strat_auroc(M, y, strata)
    thr = perm_fwer(groups, y, strata, 100)
    return {"n": len(succ), "max_abs_auroc_minus_half": round(float(np.abs(auc - 0.5).max()), 3),
            "fwer95": round(thr, 3), "n_passing": int((np.abs(auc - 0.5) > thr).sum()),
            "pass": int((np.abs(auc - 0.5) > thr).sum()) <= 1}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="10,16")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--checks-only", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    proc, snap = load_tokenizer()
    mean, std = load_stats()
    fast = Fast(proc, mean, std)
    eps = load_r047(); demos = load_demos()
    print(f"[e6] loaded {len(eps)} rollouts, {len(demos)} demos ({time.time()-t0:.0f}s)", flush=True)
    code_state(a.out)
    prov = {"tag": TAG, "tokenizer": TOK_REPO, "tokenizer_snapshot": snap,
            "tokenizer_config": {"scale": proc.scale, "min_token": proc.min_token,
                                 "vocab_size": proc.vocab_size, "time_horizon": proc.time_horizon},
            "normalisation": "MEAN_STD, lerobot/libero meta/stats.json action mean/std (all 7 dims)",
            "stats_path": STATS[0], "stats_mean": mean.tolist(), "stats_std": std.tolist(),
            "inputs_sha256": {R047: sha(R047), DEMO: sha(DEMO), STATS[0]: sha(STATS[0]),
                              os.path.join(snap, "tokenizer.json"): sha(os.path.join(snap, "tokenizer.json"))},
            "args": vars(a)}
    checks = run_checks(eps, demos, fast, proc)
    s = a.stride
    res = {"provenance": prov, "configs": {}}
    for W in [int(x) for x in a.windows.split(",")]:
        for arm in (False, True):
            name = f"W{W}_s{s}_{'arm' if arm else 'all'}"
            for e in eps + demos:
                e["_W"] = W
            tok, tstat = tokenise(eps, fast, W, s, arm)
            dtok, dstat = tokenise(demos, fast, W, s, arm)
            r = {"tokenise_r047": tstat, "tokenise_demos": dstat}
            r["a_fail_vs_success_matched"] = section_a(eps, tok, True)
            r["a_fail_vs_success_whole_episode_CONFOUNDED"] = section_a(eps, tok, False)
            r["b_anchor"] = section_b(eps, tok)
            r["c_phase_r047"] = section_c(eps, tok)
            r["c_phase_demos_cmd_rules"] = section_c(demos, dtok)
            r["e_boundary_r047_phase"] = section_e(eps, tok, "phase", s)
            r["e_boundary_r047_gripper_cmd"] = section_e(eps, tok, "cmd", s)
            r["e_boundary_demos_gripper_cmd"] = section_e(demos, dtok, "cmd", s)
            rv = Counter(t for v in tok.values() for ids in v["ids"] for t in ids)
            dv = {t for v in dtok.values() for ids in v["ids"] for t in ids}
            r["demo_vocab_coverage_of_rollout_tokens"] = round(sum(c for t, c in rv.items() if t in dv) / sum(rv.values()), 3)
            report = [x["feature"] for x in r["a_fail_vs_success_matched"]["top"][:5]]
            report += [x["feature"] for x in r["b_anchor"].get("failing_vs_successful_onset", {}).get("top", [])[:5]]
            for p in r["c_phase_r047"]["enrichment"].values():
                report += [x["feature"] for x in p["top"][:3]]
            r["d_decode"] = [describe_feature(f, eps, tok, fast, W, arm) for f in dict.fromkeys(report)]
            res["configs"][name] = r
            print(f"[e6] {name} done ({time.time()-t0:.0f}s)", flush=True)
            if W == 10 and not arm:
                checks["2_null"]["success_split"] = null_split(eps, tok)
                checks["3_planted"] = planted(eps, fast, W, s)
        if a.checks_only:
            break
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    checks["6_provenance"] = {"code_state": os.path.isdir(os.path.join(a.out, "code_state")),
                              "inputs_sha256": True}
    checks["7_plan"] = ("ROOT_SIGNAL_EXPERIMENTS E6 + prim's widened scope 10-09 (1 sliding windows, 2 a/b/c, "
                        "3 decode, 4 boundary, 5 executed actions). Not done: act_P comparison (optional; act_P is "
                        "GR00T-normalised, executed = s_d*act_P + o_d per dim, so it would need that map first), "
                        "lerobot/fast-action-tokenizer second run (no horizon, normalisation not documented).")
    json.dump(checks, open(os.path.join(a.out, "checks.json"), "w"), indent=1, default=str)
    print(json.dumps(checks, indent=1, default=str)[:4000])
    print(f"[e6] done in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
