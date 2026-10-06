"""Demonstration data: sources, the record contract, export (docs/DATA_MINER_SPEC.html).

A SOURCE turns (policy, env, perturbation) into episodes that all satisfy one
contract (`contract.py`), so the converter, the trainer and R-040's
matched-budget comparison never care where a demo came from.

  contract.py        EpisodeRecord / FrameRecord / Provenance, the manifest, G1 and G4 checks
  sources/rescue.py  S2 rescue minting (nominal chunk driven at the first forwards)
  export_lerobot.py  successes -> a LeRobot dataset in a reference dataset's schema
"""
