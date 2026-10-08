"""PoC toy: re-render the frames for the mining-spec stakeholder walkthrough.

Replays STORED actions (no policy inference) for three R-047 rollouts of scene
984 ("pick up the black bowl between the plate and the ramekin ..."), using
visualise_episode_video.replay, which also checks replay fidelity against the
stored end-effector path:

    row 14  start pose 0.2 rad, dir 1, noise seed 0  -> FAILED (the toy case)
    row 20  same start pose,           noise seed 1  -> succeeded
    row  0  nominal start (0 rad),     noise seed 0  -> succeeded

Rows are selected by LINE INDEX in runs/r047/rollouts.jsonl, not by rollout_id:
the id is identical across noise seeds (rows 14 and 20 share d46c2c057325), and
a lookup by id returns whichever comes first. The expected id, perturbation,
seed outcome and step count are asserted before replaying.

    PYTHONPATH=third_party/LIBERO-plus LIBERO_CONFIG_PATH=third_party/libero-plus-config \
    MUJOCO_GL=egl .venvs/libero-plus/bin/python experiments/poc_toy_render.py [--rows 14,20,0] [--dry]

Outputs, runs/poc_toy/render/: row<i>_<cam>_t<step>.png at the key steps, one
row<i>.mp4 per episode (both cameras side by side, the policy's orientation),
and fidelity.json. --dry checks the selection only (CPU, no env).
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from vla_harness.schema import rollout_from_dict

RUN = os.path.join(ROOT, "runs/r047/rollouts.jsonl")
OUT = os.path.join(ROOT, "runs/poc_toy/render")
EXPECT = {  # row: (rollout_id, perturbation, success, env_steps)
    14: ("d46c2c057325", {"joint_dir_seed": 1, "joint_radius_rad": 0.2}, False, 280),
    20: ("d46c2c057325", {"joint_dir_seed": 1, "joint_radius_rad": 0.2}, True, 70),
    0: ("1ea60708ae6e", {}, True, 76),
    # the one success the D14 search returned (0.4 rad, dir 2, noise seed 1);
    # rendered only with --rows 34
    34: ("8c439b0de603", {"joint_dir_seed": 2, "joint_radius_rad": 0.4}, True, 74),
}
DEFAULT_ROWS = [14, 20, 0]
# Steps for stills: start, approach, grasp onset (31 / 36 nominal), lift or slip,
# and for the failure the re-grasps and the end.
STILLS = {14: [0, 16, 31, 44, 50, 54, 100, 220, 279],
          20: [0, 16, 31, 44, 50, 60, 69],
          0: [0, 16, 36, 50, 60, 75],
          34: [0, 16, 34, 44, 54, 73]}


def load_rows(want):
    rows = [json.loads(l) for l in open(RUN)]
    out = {}
    for i in want:
        rid, pert, ok, n = EXPECT[i]
        d = rows[i]
        got = (d["rollout_id"], d["perturbation"], d["success"], d["env_steps"])
        if got != (rid, pert, ok, n):
            sys.exit(f"row {i}: expected {(rid, pert, ok, n)}, got {got}")
        out[i] = rollout_from_dict(d)
    return out


def main():
    want = DEFAULT_ROWS
    if "--rows" in sys.argv:
        want = [int(x) for x in sys.argv[sys.argv.index("--rows") + 1].split(",")]
    rows = load_rows(want)
    print("selection OK:", {i: (r.rollout_id, r.success, len(r.steps)) for i, r in rows.items()})
    if "--dry" in sys.argv:
        return
    import imageio.v2 as imageio
    from visualise_episode_video import replay, encode_video

    os.makedirs(OUT, exist_ok=True)
    fid = {}
    for i, r in rows.items():
        cams, fidelity = replay(r)
        fid[i] = {**fidelity, "rollout_id": r.rollout_id, "success": r.success,
                  "n_frames": len(cams[0]) if cams else 0, "n_cameras": len(cams)}
        print(f"row {i}: {fid[i]}")
        names = ["agent", "wrist"][:len(cams)]
        for t in STILLS[i]:
            for name, frames in zip(names, cams):
                if t < len(frames):
                    imageio.imwrite(os.path.join(OUT, f"row{i}_{name}_t{t:03d}.png"), frames[t])
        encode_video(cams, os.path.join(OUT, f"row{i}.mp4"))
    fp = os.path.join(OUT, "fidelity.json")
    old = json.load(open(fp)) if os.path.exists(fp) else {}
    old.update({str(k): v for k, v in fid.items()})
    json.dump(old, open(fp, "w"), indent=1)


if __name__ == "__main__":
    main()
