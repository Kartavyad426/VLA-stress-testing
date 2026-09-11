"""Phase 0 — supply-side coverage analysis of the LIBERO demonstrations.

Measures what the training distribution actually contains, so we can PREDICT
where the policy's operating envelope breaks before measuring it. A prediction
committed before the experiment is evidence; the same story told afterwards is
not. See PLAN.md section 3.

CPU only. No GPU, no policy, no simulator.
"""
import glob, json, sys
import numpy as np
import pandas as pd

DATA = sys.argv[1]
STATE_DIMS = ["eef_x", "eef_y", "eef_z", "rot_ax", "rot_ay", "rot_az",
              "grip_a", "grip_b"]


def load():
    files = sorted(glob.glob(f"{DATA}/data/chunk-*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tasks = pd.read_parquet(f"{DATA}/meta/tasks.parquet")
    return df, tasks


def main():
    df, tasks = load()
    S = np.stack(df["observation.state"].values)
    A = np.stack(df["action"].values)
    print(f"frames {len(df):,}  episodes {df.episode_index.nunique():,}  "
          f"tasks {df.task_index.nunique()}")

    first = df.frame_index == 0
    S0 = S[first.values]
    print(f"\n=== INITIAL STATE COVERAGE  (n={len(S0)} episodes) ===")
    print(f"{'dim':8} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'range':>9}")
    init = {}
    for i, name in enumerate(STATE_DIMS):
        c = S0[:, i]
        init[name] = dict(mean=float(c.mean()), std=float(c.std()),
                          min=float(c.min()), max=float(c.max()))
        print(f"{name:8} {c.mean():9.4f} {c.std():9.4f} {c.min():9.4f} "
              f"{c.max():9.4f} {c.max()-c.min():9.4f}")

    # translation spread is the axis LIBERO-plus perturbs as "robot initial state"
    xyz = S0[:, :3]
    centre = xyz.mean(0)
    radial = np.linalg.norm(xyz - centre, axis=1)
    print(f"\ninitial eef radial spread from mean (m): "
          f"p50={np.percentile(radial,50):.4f} p95={np.percentile(radial,95):.4f} "
          f"max={radial.max():.4f}")

    print(f"\n=== EPISODE LENGTHS ===")
    L = df.groupby("episode_index").size()
    print(f"mean {L.mean():.0f}  p50 {L.median():.0f}  p95 "
          f"{L.quantile(.95):.0f}  min {L.min()}  max {L.max()}")

    print(f"\n=== PER-TASK EPISODE COUNTS ===")
    per = df.groupby("task_index").episode_index.nunique()
    print(f"tasks {len(per)}  demos/task: min {per.min()} median "
          f"{int(per.median())} max {per.max()}")

    print(f"\n=== ACTION DISTRIBUTION (7-dim: 6 eef delta + gripper) ===")
    names = ["dx","dy","dz","droll","dpitch","dyaw","gripper"]
    for i, n in enumerate(names):
        c = A[:, i]
        print(f"{n:8} mean {c.mean():+7.4f} std {c.std():6.4f} "
              f"min {c.min():+7.3f} max {c.max():+7.3f} "
              f"|frac at rail| {np.mean(np.abs(c) > 0.99):.3f}")

    out = {"n_episodes": int(df.episode_index.nunique()),
           "n_frames": int(len(df)),
           "n_tasks": int(df.task_index.nunique()),
           "initial_state": init,
           "initial_eef_radial_p95_m": float(np.percentile(radial, 95)),
           "episode_len": {"mean": float(L.mean()), "p50": float(L.median()),
                           "max": int(L.max())},
           "camera_pose_variation": None,   # see the report: there is none
           }
    with open("experiments/phase0/coverage.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote experiments/phase0/coverage.json")


if __name__ == "__main__":
    main()
