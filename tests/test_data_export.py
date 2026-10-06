"""The data contract, G1/G4 at the record level, and the LeRobot export (CPU).

A synthetic rescue record (360x360 PNGs with a marked corner, binarised
gripper actions, 16 driven frames) goes through `export_lerobot.export` and
is read back with LeRobotDataset. The checks are the ones R-054's G1 and spot
check rely on: schema identical to the reference, state/action equal to the
adapter's mapping of the record, 180-degree orientation, truncation, stats
copied rather than recomputed, and G4 refusal.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from vla_harness.data.contract import (EpisodeRecord, EpisodeStore, FrameRecord, HeldOut,
                                       HeldOutViolation, Provenance, check_record)

REF = "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"
TASK = "pick up the black bowl next to the cookie box and place it on the plate"


def _quat(rng):
    q = rng.standard_normal(4); q /= np.linalg.norm(q)
    return [float(x) for x in q]


def make_record(root, eid="ep0", n=40, success=True, dir_seed=1000, size=360, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    st = EpisodeStore(str(root))
    img_dir = st.image_dir(eid); os.makedirs(img_dir, exist_ok=True)
    frames = []
    for t in range(n + 1):
        imgs = {}
        for cam in ("image", "image2"):
            im = np.full((size, size, 3), 40 + t % 50, np.uint8)
            im[:20, :20] = (255, 0, 0)                      # raw top-left marker
            p = os.path.join(img_dir, f"t{t:04d}_{cam}.png"); Image.fromarray(im).save(p)
            imgs[cam] = os.path.relpath(p, st.root)
        act = None if t == n else [float(x) for x in rng.uniform(-0.9, 0.9, 6)] + [1.0 if t > 20 else -1.0]
        frames.append(FrameRecord(t=t, images=imgs,
                                  proprio={"eef_pos": [float(x) for x in rng.normal(0, 0.1, 3)],
                                           "eef_quat": _quat(rng), "gripper_qpos": [0.039, -0.039],
                                           "joint_pos": [0.0] * 7},
                                  action=act, driven=(act is not None and t < 16)))
    prov = Provenance(source="rescue", policy_id="lerobot:test@0", axis="joint_radius_rad", magnitude=0.25,
                      scene_task_id=1201, suite="libero_spatial", env_id="libero_plus:libero_spatial@x",
                      env_seed=0, noise_seed=0, privileged=False, dir_seed=dir_seed, instance_id=eid, band="0.2-0.3")
    rec = EpisodeRecord(episode_id=eid, provenance=prov, instruction=TASK, frames=frames,
                        outcome={"success": success, "closest_approach_m": 0.04}, exportable=success,
                        image_meta={"size": size, "orientation": "env_raw"})
    st.write(rec)
    return st, rec


def test_record_roundtrip_and_g1(tmp_path):
    st, rec = make_record(tmp_path)
    back = st.load("ep0")
    assert back.to_dict() == rec.to_dict()
    assert check_record(back, st, cameras=("image", "image2"), action_dim=7,
                        proprio_keys=("eef_pos", "eef_quat", "gripper_qpos")) == []
    assert sum(f.driven for f in back.frames) == 16
    bad = st.load("ep0"); bad.frames[3].proprio["_gt_object_pos"] = [0]; bad.frames[4].action[0] = 1.5
    errs = check_record(bad, st, action_dim=7)
    assert any("I2" in e for e in errs) and any("outside" in e for e in errs)
    bad.provenance.noise_seed = None
    assert any("provenance missing" in e for e in check_record(bad))


def test_failure_is_kept_but_not_exportable(tmp_path):
    st, _ = make_record(tmp_path, "ok", success=True)
    make_record(tmp_path, "bad", success=False, seed=1)
    assert [r["exportable"] for r in st.rows()] == [True, False]
    assert [r.episode_id for r in st.records(exportable_only=True)] == ["ok"]


def test_heldout_refusal(tmp_path):
    p = tmp_path / "held.json"
    json.dump({"instances": [{"scene_task_id": 1201, "dir_seed": 100}]}, open(p, "w"))
    h = HeldOut([str(p)])
    with pytest.raises(HeldOutViolation):
        h.refuse(1201, 100)
    h.refuse(1201, 1000); h.refuse(984, 100)
    assert h.violations([{"episode_id": "e", "scene_task_id": 1201, "dir_seed": 100}])


def test_adapter_gripper_and_orientation():
    pytest.importorskip("cv2")
    from vla_harness.data.export_lerobot import LiberoIpecAdapter
    ad = LiberoIpecAdapter()
    a = ad.action([0.1, -0.2, 0.3, 0.01, 0.02, 0.03, -1.0])
    assert a[6] == 1.0 and np.allclose(a[:6], [0.1, -0.2, 0.3, 0.01, 0.02, 0.03])
    assert ad.action([0] * 6 + [1.0])[6] == 0.0
    assert np.array_equal(ad.inverse_action(a), np.float32([0.1, -0.2, 0.3, 0.01, 0.02, 0.03, -1.0]))
    with pytest.raises(ValueError):
        ad.action([0] * 6 + [0.3])
    im = np.zeros((360, 360, 3), np.uint8); im[:36, :36] = 255
    out = ad.image(im)
    assert out.shape == (256, 256, 3) and out[-5:, -5:].min() == 255 and out[:5, :5].max() == 0


def _need_reference():
    lr = pytest.importorskip("lerobot")
    from lerobot.utils.constants import HF_LEROBOT_HOME
    if not os.path.exists(os.path.join(HF_LEROBOT_HOME, REF, "meta", "info.json")):
        pytest.skip("reference dataset not converted locally")


def test_export_matches_reference_schema(tmp_path):
    _need_reference()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from vla_harness.data.export_lerobot import LiberoIpecAdapter, export, reference_info
    st, rec = make_record(tmp_path / "rec", "ep0", n=40)
    make_record(tmp_path / "rec", "ep1", n=30, seed=2, dir_seed=1001)
    make_record(tmp_path / "rec", "fail", n=30, seed=3, success=False, dir_seed=1002)
    out = tmp_path / "ds"
    s = export([str(tmp_path / "rec")], str(out), "local/test_export", REF,
               "nvidia/gr00t17-lerobot-libero_spatial-640", truncate_steps=24)
    assert s["episodes"] == 2 and s["frames"] == 48 and s["unknown_tasks"] == []
    ds = LeRobotDataset("local/test_export", root=str(out), video_backend="pyav")
    ref = reference_info(REF)
    for k in ("observation.images.image", "observation.images.wrist_image", "observation.state", "action"):
        a, b = dict(ds.meta.features[k]), dict(ref["features"][k])
        a.pop("info", None); b.pop("info", None)
        a["shape"], b["shape"] = list(a["shape"]), list(b["shape"])
        assert a == b, k
    assert ds.fps == ref["fps"] == 20
    ad = LiberoIpecAdapter()
    for i in range(24):
        x = ds[i]
        f = rec.frames[i]
        assert np.array_equal(x["observation.state"].numpy(), ad.state(f.proprio))
        assert np.array_equal(x["action"].numpy(), ad.action(f.action))
        assert int(x["frame_driven"]) == int(f.driven) and int(x["frame_source_t"]) == f.t
        assert x["task"] == TASK
    img = x["observation.images.image"]
    img = img.permute(1, 2, 0).numpy()
    assert img.shape == (256, 256, 3)
    # the raw top-left red marker must sit bottom-right after the 180-degree flip (lossy video: tolerant)
    assert img[-8:, -8:, 0].mean() > 0.7 and img[:8, :8, 0].mean() < 0.4
    stats = json.load(open(out / "meta" / "stats.json"))
    assert stats["action"]["max"][6] == 1.0 and abs(stats["action"]["mean"][0] - 0.1531) < 1e-3
    prov = [json.loads(l) for l in open(out / "meta" / "provenance.jsonl")]
    assert [p["episode_id"] for p in prov] == ["ep0", "ep1"] and prov[0]["n_driven"] == 16


def test_export_refuses_heldout(tmp_path):
    _need_reference()
    from vla_harness.data.export_lerobot import export
    make_record(tmp_path / "rec", "ep0", n=10, dir_seed=100)
    p = tmp_path / "held.json"
    json.dump({"instances": [{"scene_task_id": 1201, "dir_seed": 100}]}, open(p, "w"))
    with pytest.raises(RuntimeError, match="G4"):
        export([str(tmp_path / "rec")], str(tmp_path / "ds"), "local/x", REF,
               "nvidia/gr00t17-lerobot-libero_spatial-640", heldout_lists=[str(p)])
