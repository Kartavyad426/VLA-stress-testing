"""Collect lerobot-eval results and compare against the published target.

Reads whatever eval json lerobot wrote, prints per-suite success with a Wilson
CI at n=100 (10 tasks x 10 episodes), and the gap to published.
"""
import json, sys, pathlib, math

PUBLISHED = {"libero_spatial": 90.0, "libero_object": 96.0,
             "libero_goal": 92.0, "libero_10": 71.0}
PRIOR_3264 = {"libero_spatial": 63.0, "libero_object": 93.0,
              "libero_goal": 81.0, "libero_10": 56.0}


def wilson(k, n, z=1.96):
    if not n: return (0.0, 1.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))


def find_results(run_dir):
    out = {}
    for p in pathlib.Path(run_dir).rglob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        # lerobot writes per-suite aggregates; be liberal about shape
        for key in ("aggregated", "eval", "results"):
            if isinstance(d, dict) and key in d:
                out[p.as_posix()] = d
                break
        else:
            if isinstance(d, dict) and any("success" in str(k).lower() for k in d):
                out[p.as_posix()] = d
    return out


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "experiments/repro/runs"
    res = find_results(run)
    if not res:
        print(f"no result json under {run}; inspect eval.log directly")
        sys.exit(1)
    for path, d in res.items():
        print(f"\n--- {path}")
        print(json.dumps(d, indent=2)[:2000])
