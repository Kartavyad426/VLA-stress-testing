"""Run a command and record its memory peaks: summed and largest-process RSS (and anonymous RSS) over the
whole process tree, and nvidia-smi's used GPU memory (whole GPU; jobs hold the GPU flock). Polls every 2 s.

  .venvs/groot/bin/python experiments/stepmeter.py <peaks.json> -- cmd args...

Exits with the command's return code; writes {rc, wall_s, rss_tree_max_gib, rss_anon_tree_max_gib,
rss_proc_max_gib, gpu_used_max_gib, gpu_used_start_gib} to peaks.json.
"""
import json
import subprocess
import sys
import threading
import time

import psutil


def main():
    out, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
    t0 = time.time()
    p = subprocess.Popen(cmd)
    peak = {"rss_tree_max_gib": 0.0, "rss_anon_tree_max_gib": 0.0, "rss_proc_max_gib": 0.0, "gpu_used_max_gib": 0.0}

    def gpu():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            return int(r.stdout.split()[0]) / 1024
        except Exception:
            return None
    peak["gpu_used_start_gib"] = gpu()
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                root = psutil.Process(p.pid)
                procs = [root] + root.children(recursive=True)
                tot = anon = big = 0.0
                for q in procs:
                    try:
                        m = q.memory_full_info() if hasattr(q, "memory_full_info") else q.memory_info()
                        rss = m.rss / 2**30
                        tot += rss; big = max(big, rss)
                        anon += (m.rss - getattr(m, "shared", 0)) / 2**30
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak["rss_tree_max_gib"] = max(peak["rss_tree_max_gib"], tot)
                peak["rss_anon_tree_max_gib"] = max(peak["rss_anon_tree_max_gib"], anon)
                peak["rss_proc_max_gib"] = max(peak["rss_proc_max_gib"], big)
            except psutil.NoSuchProcess:
                pass
            g = gpu()
            if g is not None:
                peak["gpu_used_max_gib"] = max(peak["gpu_used_max_gib"], g)
            stop.wait(2)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    rc = p.wait()
    stop.set(); th.join(timeout=15)
    peak.update(rc=rc, wall_s=round(time.time() - t0, 1),
                note="rss_anon = rss - shared (file-backed) per process, summed; gpu = whole GPU (nvidia-smi)")
    json.dump({k: (round(v, 3) if isinstance(v, float) else v) for k, v in peak.items()}, open(out, "w"), indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
