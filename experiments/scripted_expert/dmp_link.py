"""Read-only link to the DMP PoC code (branch exp/dmp-poc, worktree ../VLA-exp-dmp-poc).

The scripted expert uses DMP's root restore, roots, executor (Runner) and keep
checks (judge) unchanged, so the two sources of corrective demos are scored the
same way (user, 10-20: "use from DMP"). Nothing here is copied or edited; the
files are imported from DMP's worktree and must be identical to the commit the
comparison is pinned to, DMP_PIN. If DMP changes one of them, imports refuse
until the pin is moved on purpose.

UNVERIFIED: new numeric code, not yet checked by val.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.dirname(os.path.dirname(HERE))
DMP_WORKTREE = os.path.join(os.path.dirname(WORKTREE), "VLA-exp-dmp-poc")
DMP_CODE = os.path.join(DMP_WORKTREE, "experiments/dmp_poc")
DMP_PIN = "dae447a"
USED = ("restore.py", "roots.py", "execute.py", "demos.py", "dmp.py")   # execute.py imports dmp.py


def check_pin():
    files = [f"experiments/dmp_poc/{f}" for f in USED]
    r = subprocess.run(["git", "-C", DMP_WORKTREE, "diff", "--quiet", DMP_PIN, "--", *files])
    if r.returncode != 0:
        raise RuntimeError(f"DMP files {USED} differ from {DMP_PIN} in {DMP_WORKTREE}; "
                           "re-pin deliberately (dmp_link.DMP_PIN) before comparing")


check_pin()
if DMP_CODE not in sys.path:
    sys.path.insert(0, DMP_CODE)

import demos  # noqa: E402,F401
import execute  # noqa: E402,F401
import restore  # noqa: E402,F401
import roots  # noqa: E402,F401
