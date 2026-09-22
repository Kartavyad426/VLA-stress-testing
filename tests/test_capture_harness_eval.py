"""Capture must be reachable from the PRODUCTION run path, not just the smoke.

The smoke script builds its own policy. A capture campaign runs through
experiments/harness_eval.py, so the flag has to exist there or the captured
runs would be produced by a different code path than every other run on disk.
"""
from __future__ import annotations

import subprocess
import sys


def test_harness_eval_exposes_capture_flags():
    out = subprocess.run([sys.executable, "experiments/harness_eval.py", "--help"],
                         capture_output=True, text=True).stdout
    assert "--capture-dir" in out
    assert "--capture-k-resample" in out
