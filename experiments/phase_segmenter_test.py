"""Regression test for LiberoPhaseSegmenter, against stored traces.

Covers the three defects recorded as O1-O3 in IMPLEMENTATION_OVERSIGHTS.md.
Runs on CPU over `runs/camp_20260916-0015_*`; no policy, no GPU.

    python3 experiments/phase_segmenter_test.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.schema import rollout_from_dict
from vla_harness.mining.phases_libero import LiberoPhaseSegmenter
from vla_harness.mining.phases import TRANSPORT

RUNS = "runs/camp_20260916-0015_{}"
SUITES = ("libero_spatial", "libero_object", "libero_10")


def load(suite, limit=None):
    out = []
    for i, line in enumerate(open(f"{RUNS.format(suite)}/rollouts.jsonl")):
        if limit and i >= limit:
            break
        out.append(rollout_from_dict(json.loads(line)))
    return out


def t_o1_target_is_bddl_first(data):
    """O1: the target is the BDDL manipuland, not the alphabetically first name."""
    seg, bad = LiberoPhaseSegmenter(), 0
    for suite in SUITES:
        for r in data[suite]:
            d = r.steps[0].obs_state.get("_gt_eef_to_object") or {}
            if not d:
                continue
            if seg._target(r.steps[0].obs_state) != next(iter(d)):
                bad += 1
    return bad == 0, f"{bad} episodes segment against a non-BDDL-first object"


def t_o2_lift_not_gated_by_closure(data):
    """O2: an object that rose off the table counts as held, whatever the aperture.

    Scored on libero_spatial successes, where the object provably left the table.
    """
    seg = LiberoPhaseSegmenter()
    hit = tot = 0
    for r in data["libero_spatial"]:
        if not r.success:
            continue
        tot += 1
        tgt = seg._target(r.steps[0].obs_state)
        z0 = (r.steps[0].obs_state.get("_gt_object_pos", {}).get(tgt) or [0, 0, 0])[2]
        if any(((s.obs_state.get("_gt_object_pos", {}).get(tgt) or [0, 0, z0])[2] - z0)
               > seg.lift_m and seg._holding(r, i)
               for i, s in enumerate(r.steps)):
            hit += 1
    return hit == tot, f"lift route reaches {hit}/{tot} spatial successes"


def t_o3_travel_transient_is_not_a_grasp(data):
    """O3: a closure onto empty air must not register as holding BY GRIPPER.

    A closure is 'onto empty air' when the fingers do reach full closure while
    the command is still held -- nothing was between them, by definition.
    """
    seg = LiberoPhaseSegmenter()
    false_fires = tot = 0
    for suite in SUITES:
        for r in data[suite]:
            steps = r.steps
            cc = [seg._commanded_closed(s) for s in steps]
            prev = False
            for i, c in enumerate(cc):
                if c and not prev:
                    k, closed_at = i, None
                    while k < len(cc) and cc[k]:
                        if seg._closed(steps[k].obs_state):
                            closed_at = k
                            break
                        k += 1
                    if closed_at is not None:       # closed on empty air
                        tot += 1
                        # the GRIPPER branch alone. Scoring `_holding` here
                        # would count lift fires too -- and an object already
                        # held during a re-grasp legitimately lifts, so the OR
                        # is not the thing this test is about.
                        if any(seg._holding_by_gripper(r, j)
                               for j in range(i, closed_at)):
                            false_fires += 1
                prev = c
    # Not zero. ~1.5% of empty-air closures still stall briefly mid-travel --
    # the fingers decelerate, pause within tolerance, then continue. Tightening
    # `stall_m` to 0.001 halves it but costs libero_10 success coverage, so the
    # residual is accepted and BOUNDED here rather than tuned away.
    rate = false_fires / max(1, tot)
    return rate < 0.03, (f"{false_fires}/{tot} = {rate*100:.1f}% empty-air "
                         f"closures read as holding (bound: 3%)")


def t_successes_show_transport(data):
    """The standing validity check: every success must contain a TRANSPORT phase."""
    seg, msgs, ok = LiberoPhaseSegmenter(), [], True
    for suite in SUITES:
        hit = tot = 0
        for r in data[suite]:
            if not r.success:
                continue
            tot += 1
            segs, _ = seg(r)
            hit += any(s.phase == TRANSPORT for s in segs)
        msgs.append(f"{suite}={hit}/{tot}")
        ok &= hit == tot
    return ok, ", ".join(msgs)


def t_branches_scored_separately(data):
    """O4 (OPEN): score each holding route ALONE, never via the OR.

    A disjunction cannot be validated by asking whether the disjunction fired.
    This is reported, not asserted: the branches are NOT expected to agree.
    `by_gripper` needs the held object to be thicker than `closed_m`, which is
    false for thin objects (libero_spatial bowl rims), and `by_lift` needs
    privileged simulator state. They cover different regimes on purpose.

    What to watch: `by_gripper` is the route a REAL robot could run. Its
    coverage here, scored against `by_lift`, is the only evidence we have about
    whether that route transfers.
    """
    seg, msgs, ok = LiberoPhaseSegmenter(), [], True
    for suite in SUITES:
        g = l = tot = 0
        for r in data[suite]:
            if not r.success:
                continue
            tot += 1
            tgt = seg._target(r.steps[0].obs_state)
            z0 = (r.steps[0].obs_state.get("_gt_object_pos", {}).get(tgt) or [0, 0, 0])[2]
            g += any(seg._holding_by_gripper(r, i) for i in range(len(r.steps)))
            l += any(seg._holding_by_lift(r, i, tgt, z0) for i in range(len(r.steps)))
        msgs.append(f"{suite} gripper={g}/{tot} lift={l}/{tot}")
        ok &= (g == tot and l == tot)
    return ok, "; ".join(msgs)


TESTS = [("O1 target is BDDL-first", t_o1_target_is_bddl_first),
         ("O2 lift not gated by closure", t_o2_lift_not_gated_by_closure),
         ("O3 travel transient is not a grasp", t_o3_travel_transient_is_not_a_grasp),
         ("-- successes show TRANSPORT", t_successes_show_transport)]

# Reported every run, never asserted. See the docstring for why.
REPORTS = [("O4 branch coverage (open)", t_branches_scored_separately)]

if __name__ == "__main__":
    data = {s: load(s) for s in SUITES}
    print(f"loaded {sum(len(v) for v in data.values())} traces\n")
    failed = 0
    for name, fn in TESTS:
        try:
            ok, detail = fn(data)
        except AttributeError as e:
            ok, detail = False, f"not implemented: {e}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}")
        failed += not ok
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed\n")
    for name, fn in REPORTS:
        _, detail = fn(data)
        print(f"  [----] {name:38s} {detail}")
    sys.exit(1 if failed else 0)
