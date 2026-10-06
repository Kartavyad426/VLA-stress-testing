"""Where does the load's RSS peak happen? Samples RSS every 20 ms with the main thread's stack. CPU only."""
import sys, os, threading, time, traceback
sys.path.insert(0, os.getcwd())
def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 2**20
best = {"rss": 0, "stack": None}; marks = []
def parts():
    d = {l.split(":")[0]: int(l.split()[1]) / 2**20 for l in open("/proc/self/status") if l.startswith(("RssAnon", "RssFile", "RssShmem"))}
    return d
panon = {"v": 0, "at": None}
main = threading.main_thread().ident
def sampler():
    last = 0
    while True:
        r = rss(); pa = parts()
        if pa["RssAnon"] > panon["v"]:
            panon["v"] = pa["RssAnon"]; panon["at"] = (round(r, 2), {k: round(v, 2) for k, v in pa.items()},
                " <- ".join(f"{os.path.basename(x.filename)}:{x.lineno}:{x.name}" for x in traceback.extract_stack(sys._current_frames()[main])[-5:][::-1]))
        if r > best["rss"]:
            best["rss"] = r; best["stack"] = traceback.format_stack(sys._current_frames()[main])
        if abs(r - last) > 1.0:
            f = traceback.extract_stack(sys._current_frames()[main])
            marks.append((round(r, 1), " <- ".join(f"{os.path.basename(x.filename)}:{x.lineno}:{x.name}" for x in f[-6:][::-1])))
            last = r
        time.sleep(0.02)
threading.Thread(target=sampler, daemon=True).start()
MODE = sys.argv[1] if len(sys.argv) > 1 else "new"
if MODE == "old":
    import vla_harness.policies.lerobot_policy as LP
    def _old(policy_cls, checkpoint, cfg, cast):
        p = policy_cls.from_pretrained(checkpoint, config=cfg)
        return p.to(cast) if cast is not None else p
    LP._from_pretrained_cast = _old
print("MODE", MODE)
sys.argv = [sys.argv[0]]
from experiments.r044_reverse import build_policy
pol = build_policy(); pol.device = "cpu"; pol._load()
time.sleep(0.2)
print("PEAK %.2f GiB at:" % best["rss"]); print("".join(best["stack"][-14:]))
print("peak total RSS parts at end:", parts())
print("PEAK ANONYMOUS RSS %.2f GiB (total RSS then, parts, where):" % panon["v"], panon["at"])
