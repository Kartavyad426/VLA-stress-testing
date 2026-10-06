R-056 / R-058 post-mortem analysis (CPU only, existing artifacts). Report: docs/R056_POSTMORTEM.html
Run from this folder:
  ../../.venvs/groot/bin/python perstart.py rows.pkl      # paired per-start table (joins manifest/rollouts BY POSITION: rollout_id ignores the noise seed)
  ../../.venvs/groot/bin/python an1.py ... an9.py          # read rows.pkl
  (cd ../.. && .venvs/libero-plus/bin/python experiments/analysis_r056_postmortem/padcheck.py)   # action_is_pad -> action_mask
  ../../.venvs/libero-plus/bin/python decomp.py            # loss-by-source attempt (inconclusive, R2 ~ 0)
