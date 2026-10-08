# R-060 G0c amendment: DRAFT, not applied (impl, 2026-10-08 18:31 IST)

For SFT and my prim to review; the user decides. Nothing in experiments/r060_gates.py or
vla_harness/ has been changed. Evidence: runs/r060/gates/g0c_diag*.json (scripts
experiments/r060_g0c_diag*.py), all CPU/EGL render, no model.

## Why the gate as written cannot pass (measured)

1. The reference frame is the wrong one. robosuite samples camera observables on a timer
   (robosuite/utils/observables.py:224-240). A step's returned frame shows the state after
   substep 23 of 25, not the final state (g0c_diag_substep.json, clean env, 4 of 6 steps
   bitwise; the other 2 are within 2-6 px of EGL jitter). `perturbed_frames` / `nominal_frames`
   re-render the current state (sim.forward + force_update). So "re-render vs reset/step
   frame" differs on every row: agent 110-430 px, wrist 1.4k-9k px.
2. Reusing the control env breaks rows. A control env that is reset again (scene 1132)
   drifts about 3e-9 in qpos from a fresh one and renders 35k-57k px differently (g0c_diag_c3/c4/c5).
   The variant is bitwise equal to a fresh control. All 3/25 phys failures are this.
   Separately, deleting any other env corrupts a surviving env's render (~all pixels) until
   its next reset (c5).
3. EGL render jitter: repeated renders of one state give up to 6 px different (2 distinct
   of 10). Motion blur carries it (noise 6: 25-40 px, ≤1 level). The blur itself is
   deterministic (20/20 identical, any thread count).

## Proposed gate changes

- A1. Compare re-render vs re-render on both sides (forward + force_update), never a
  step/reset frame. With this, measured P: camera 13/14 bitwise (one 2 px), noise 7/8
  bitwise (motion 6: 25 px).
- A2. A fresh control env and a fresh variant env per row, each reset once. Nothing is
  deleted between that reset and the renders.
- A3. A tolerance for EGL jitter, replacing bitwise: n_diff ≤ 8 px per camera at max_abs ≤ 1,
  or ≤ 8 px at any max_abs when the pixel set changes between two repeated renders
  (to be agreed). Or keep bitwise and render each side 3× and take the mode.
- A4. Replace the noise end-to-end check (variant's own reset frame) with: noise of the
  variant env's clean re-render vs control perturbed_frames, under the same seed.

## Not a gate change: a code bug in the N side (needs the user's go)

`nominal_frames()` puts LIBERO's original agentview pose (nominal._CAMERA:
pos 0.658613/0/1.610350) in place for LIBERO-Plus camera variants. The control env
actually renders LIBERO-Plus's canonical pose, rounded to 4 dp (0.6586/0/1.6104; quat
differs by 3.4e-5). Result: N vs control re-render differs by 21.8k-24.2k px on 14/14
camera rows (max 52). Writing `_bench_camera_pose(0,0,1.0)` instead gives 13/14 bitwise
(one 6 px wrist jitter). Noise rows are unaffected (8/8 bitwise). Fix: in nominal_frames,
for libero_plus camera variants, use `self._bench_camera_pose(0, 0, 1.0)`.

## Side effect to account for in pair generation

Each force_update re-render moves the sampling phase of later step frames
(substep 23 → 21, 17, 15, 13 with two paired calls per step). Physics stays bitwise
unchanged over 40 steps open-loop, but every later step frame differs from an
unpaired run's. Closed loop, the policy's own frames, and so its actions, change.
