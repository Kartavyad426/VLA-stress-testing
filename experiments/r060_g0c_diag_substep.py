"""R-060 G0c diagnosis, follow-up to r060_g0c_diag.py (step a): WHICH substep's state does
the frame returned by an env step show? robosuite samples camera observables on a timer
(robosuite/utils/observables.py:224-240), not after the last substep. For each env step,
every substep's post-step state is re-rendered (set_state + forward) and compared with the
step frame. Done twice: in a clean env, and in one where nominal_frames() +
perturbed_frames() (each a force_update) are called before every step, as the paired-N
runs do. CPU/render only. Writes runs/r060/gates/g0c_diag_substep.json.

  env $ENV flock /tmp/vla_gpu.lock .venvs/libero-plus/bin/python experiments/r060_g0c_diag_substep.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))

import numpy as np

from r060_g0c_diag import log, parts, px  # noqa: E402
from r060_gates import OUT, make_env  # noqa: E402

VAR = "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_22_15_100_0_0_initstate_0"


def matches(env, S, states):
    """indices k such that the forward render of states[k] equals S on both cameras"""
    sim = parts(env)[3]
    cur = sim.get_state()
    hit, nd = [], []
    for k, st in enumerate(states):
        sim.set_state(st); sim.forward()
        R = px(env, False)
        a = int((R["image"] != S["image"]).any(-1).sum()); w = int((R["image2"] != S["image2"]).any(-1).sum())
        nd.append([a, w])
        if a == 0 and w == 0:
            hit.append(k)
    sim.set_state(cur)          # no forward: leave xpos as the step left it
    return hit, nd


def run(paired: bool, n_steps=6):
    from vla_harness.schema import PerturbationSpec
    env = make_env(1030)
    env.reset(0, PerturbationSpec())
    outer, wrapper, inner, sim = parts(env)
    rows = []
    for t in range(n_steps):
        act = np.array([0.4 * np.sin(t / 2), 0.4 * np.cos(t / 2), 0.2, 0, 0, 0, -1], dtype=np.float64)
        if paired:
            env.nominal_frames(); env.perturbed_frames(VAR)
        states = []
        real = sim.step

        def spy(*a, **k):
            r = real(*a, **k)
            states.append(sim.get_state())       # state AFTER each substep
            return r
        sim.step = spy
        try:
            raw, *_ = wrapper.step(act)
        finally:
            sim.step = real
        S = dict(outer._format_raw_obs(raw)["pixels"])
        hit, nd = matches(env, S, states)
        # substep index is 0-based: states[k] = after substep k+1 of 25
        rows.append({"t": t, "matching_substeps_after": [k + 1 for k in hit], "n_substeps": len(states),
                     "ndiff_px_[agent,wrist]_by_substep": nd})
        log("paired" if paired else "clean", t, "frame = state after substep", [k + 1 for k in hit],
            "min ndiff", min(nd, key=sum))
    del env
    return rows


def main():
    res = {"clean": run(False), "paired_renders_before_each_step": run(True)}
    p = os.path.join(OUT, "g0c_diag_substep.json")
    json.dump(res, open(p, "w"), indent=1)
    log("wrote", p)


if __name__ == "__main__":
    main()
