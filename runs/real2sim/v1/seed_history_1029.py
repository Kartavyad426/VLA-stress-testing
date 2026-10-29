"""res31 10-29 (user: "dont remove the prev checklist result tho, append into a new section while saying what changes").
One-off: put the earlier checklist results back into each re-scored scene's v1_report.json as 'history', oldest first:
  1. <scene>/before_redact_1029/v1_report.json, exact (the phase-1 / dev build before redaction)
  2. the 14:20 redaction rerun. The 14:32 --report_only rescore overwrote it before history was kept; it is rebuilt by
     undoing that rescore's three rule changes (the only code changes between the two): drop/scale n/a without a container
     was FAIL with an empty detail; B3 with 0 probes was "pass"; A8 'needs review' for a near-built unbuilt grasp was pass.
Then build_v1 --report_only (rereport_1029.sh) re-renders the page with the history sections.
    .venvs/real2sim/bin/python runs/real2sim/v1/seed_history_1029.py <scene dir> ...
"""
import copy, json, os, sys

L_BEFORE = "first build, before redaction (phase-1 batch 10-29 / dev scenes)"
L_RERUN = "redaction rerun 10-29, 13:10 c24 / 14:20 the rest (reconstructed: rescore overwrote it; rebuilt by undoing the rescore's three rule changes)"
L_NOW = ("rescore 10-29 14:32: drops n/a without a container (was FAIL, empty detail); scripted expert with 0 probes n/a "
         "(was 'pass'); A8 'needs review' when an unbuilt grasp is within 5 cm of a built object (was pass)")
for s in sys.argv[1:]:
    cur_p, bef_p = os.path.join(s, "v1_report.json"), os.path.join(s, "before_redact_1029", "v1_report.json")
    cur = json.load(open(cur_p))
    if cur.get("history"):
        print(s, "already has history, skip")
        continue
    hist = []
    if os.path.exists(bef_p):
        b = json.load(open(bef_p))
        hist.append(dict(created=b["created"], label=L_BEFORE, rows=b["rows"]))
    rows = copy.deepcopy(cur["rows"])
    grasp = next((r for r in rows if r["id"] == "A3" and r["check"].startswith("real grasps")), None)
    for r in rows:
        if r["id"] == "A3" and r["check"].startswith("real drops") and r["state"] == "n/a" and grasp and grasp["state"] != "n/a":
            r["state"], r["detail"] = "fail", ""
        elif r["id"] == "B3" and r["state"] == "n/a" and r["detail"].startswith("0 probes"):
            r["state"], r["detail"] = "pass", "0/0 probes pass"
        elif r["id"] == "A8" and r["state"] == "needs review":
            r["state"], r["detail"] = "pass", " | ".join(r["detail"].split(" | ")[:2])
    hist.append(dict(created="2026-10-29 13:10-14:20", label=L_RERUN, rows=rows))
    cur["history"], cur["label"] = hist, L_NOW
    json.dump(cur, open(cur_p, "w"), indent=1, default=float)
    print(s, f"history: {len(hist)} entries")
