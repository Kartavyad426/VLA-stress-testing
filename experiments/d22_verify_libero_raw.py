"""D22 (user 2026-10-07, via prim): verify the raw LIBERO-Spatial demonstration HDF5s in data/libero_raw/.
CPU only.

  .venvs/libero-plus/bin/python experiments/d22_verify_libero_raw.py [--root data/libero_raw/libero_spatial]

Per file:
  * sha256, compared with the Hugging Face LFS sha256 of yifengzhu-hf/LIBERO-datasets (the source
    LIBERO-plus's download_utils uses, HF_REPO_ID);
  * the demo count, and per demo the datasets (states, actions, obs/*, ...) with shapes, plus file and demo attrs;
  * initial states: each demo's states[0] (and a per-demo `init_state` attr if present) against the LIBERO-Plus
    init files of the same task (third_party/LIBERO-plus/.../init_files/libero_spatial/<task>.init and
    .pruned_init): the nearest init row by max |diff|, how many demos match a row exactly (<= 1e-9) or closely
    (<= 1e-4), and the state dimension of each.
Writes <root>/../verify.json and <root>/../checksums.sha256.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import h5py
import numpy as np

INIT_DIR = "third_party/LIBERO-plus/libero/libero/init_files/libero_spatial"
HF_REPO = "yifengzhu-hf/LIBERO-datasets"


def sha256(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def attrs(o):
    out = {}
    for k, v in o.attrs.items():
        v = v.decode() if isinstance(v, bytes) else v
        if isinstance(v, np.ndarray):
            v = {"array_shape": list(v.shape), "dtype": str(v.dtype)}
        elif isinstance(v, (np.integer, np.floating)):
            v = v.item()
        out[k] = v if not isinstance(v, str) or len(v) < 400 else v[:400] + "...(truncated)"
    return out


def nearest(x, rows):
    if rows is None or rows.shape[1] != x.shape[0]:
        return None
    d = np.abs(rows - x[None]).max(1)
    i = int(d.argmin())
    return i, float(d[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/libero_raw/libero_spatial")
    a = ap.parse_args()
    import torch
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(HF_REPO, files_metadata=True)
    hf = {s.rfilename: {"size": s.size, "sha256": getattr(s.lfs, "sha256", None) if s.lfs is not None and not isinstance(s.lfs, dict)
                        else (s.lfs or {}).get("sha256")} for s in info.siblings if s.rfilename.startswith("libero_spatial/")}
    files = sorted(glob.glob(os.path.join(a.root, "*.hdf5")))
    out = {"source": f"huggingface:{HF_REPO}@{info.sha}", "license": (info.card_data or {}).get("license") if info.card_data else None,
           "root": a.root, "n_files": len(files), "hf_files": len(hf), "files": {}}
    sums = []
    for p in files:
        name = os.path.basename(p)
        task = name[: -len("_demo.hdf5")]
        s = sha256(p)
        sums.append(f"{s}  {p}")
        h = hf.get(f"libero_spatial/{name}", {})
        rec = {"bytes": os.path.getsize(p), "sha256": s, "sha256_matches_hf": s == h.get("sha256"),
               "size_matches_hf": os.path.getsize(p) == h.get("size")}
        inits = {}
        for ext in ("init", "pruned_init"):
            ip = os.path.join(INIT_DIR, f"{task}.{ext}")
            inits[ext] = np.asarray(torch.load(ip, weights_only=False), dtype=np.float64) if os.path.exists(ip) else None
        rec["init_files"] = {k: (list(v.shape) if v is not None else "missing") for k, v in inits.items()}
        with h5py.File(p, "r") as f:
            data = f["data"]
            demos = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k)
            rec["file_attrs"] = attrs(data)
            rec["n_demos"] = len(demos)
            d0 = data[demos[0]]
            rec["demo0_datasets"] = {}
            d0.visititems(lambda n, o: rec["demo0_datasets"].__setitem__(n, [list(o.shape), str(o.dtype)]) if isinstance(o, h5py.Dataset) else None)
            rec["demo0_attrs"] = attrs(d0)
            lens, match = [], {k: {"exact": 0, "close": 0, "nearest_maxdiff": []} for k in inits}
            init_attr_vs_states0 = []
            missing = []
            for dk in demos:
                d = data[dk]
                for need in ("states", "actions", "obs"):
                    if need not in d:
                        missing.append(f"{dk}/{need}")
                if "states" not in d:
                    continue
                st0 = np.asarray(d["states"][0], dtype=np.float64)
                lens.append(int(d["actions"].shape[0]) if "actions" in d else None)
                if "init_state" in d.attrs:
                    init_attr_vs_states0.append(float(np.abs(np.asarray(d.attrs["init_state"], dtype=np.float64) - st0).max()))
                for k, rows in inits.items():
                    nn = nearest(st0, rows)
                    if nn is None:
                        continue
                    match[k]["nearest_maxdiff"].append(nn[1])
                    match[k]["exact"] += nn[1] <= 1e-9
                    match[k]["close"] += nn[1] <= 1e-4
            rec["demo_lengths"] = {"min": min(lens), "median": float(np.median(lens)), "max": max(lens)} if lens else None
            rec["state_dim"] = int(d0["states"].shape[1]) if "states" in d0 else None
            rec["missing"] = missing
            rec["init_state_attr_vs_states0_maxdiff"] = max(init_attr_vs_states0) if init_attr_vs_states0 else "no init_state attr"
            for k, m in match.items():
                v = m.pop("nearest_maxdiff")
                m["n_compared"] = len(v)
                if v:
                    m["nearest_maxdiff_median"] = float(np.median(v))
                    m["nearest_maxdiff_min"] = float(np.min(v))
            rec["states0_vs_init_files"] = match
        out["files"][task] = rec
        print(f"{task[:60]:60s} demos {rec['n_demos']:3d} sha_ok {rec['sha256_matches_hf']} dim {rec['state_dim']} "
              f"init {rec['init_files']} match {json.dumps(rec['states0_vs_init_files'])}", flush=True)
    base = os.path.dirname(os.path.abspath(a.root))
    json.dump(out, open(os.path.join(base, "verify.json"), "w"), indent=1, default=str)
    open(os.path.join(base, "checksums.sha256"), "w").write("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
