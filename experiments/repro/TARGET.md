# Reproduction target — SmolVLA on LIBERO

**Checkpoint:** `HuggingFaceVLA/smolvla_libero`
**Protocol:** 10 episodes/task × 10 tasks × 4 suites = 400 episodes. LeRobot
recommends averaging over 3 seeds; success rates vary a few percent by seed.

## Published (SmolVLA paper, arXiv:2506.01844)

| Suite | Published |
|---|---|
| LIBERO-Spatial | ~90 |
| LIBERO-Object | ~96 |
| LIBERO-Goal | ~92 |
| LIBERO-Long (libero_10) | ~71 |
| **Average** | **~87.3** |

## Prior failed reproductions — TWO independent, both open, no maintainer reply

| | [#3264](https://github.com/huggingface/lerobot/issues/3264) — *official checkpoint, eval only* | [#3287](https://github.com/huggingface/lerobot/issues/3287) — *self-trained* | Published |
|---|---|---|---|
| Spatial | 63.0 | 83.0 | ~90 |
| Object | 93.0 | 70.0 | ~96 |
| Goal | 81.0 | 70.0 | ~92 |
| Long | 56.0 | 44.8 | ~71 |
| **Avg** | **73.25** | ~67 | **~87.3** |

#3264 is the one that bears on us: same checkpoint, evaluation only, no training.
#3287 trained its own model, so it is weaker evidence about the checkpoint but
corroborates that the published numbers are hard to hit.

## Leading hypothesis: `n_action_steps`

The published config ships `chunk_size: 50`, **`n_action_steps: 1`**.

LeRobot's own Pi0.5 reproduction page overrides this explicitly on the command
line — `--policy.n_action_steps=10`, "matching the original OpenPI
implementation" — and reproduces published numbers closely (97.5 vs 96.85 avg).

So the override is known to be necessary for at least one policy, and is *not*
in the config. If SmolVLA's published numbers were produced with chunked
execution and the config default replays one action per forward pass, the
open-loop horizon differs and behaviour differs. **`n_action_steps` is the first
variable to test**, and it costs a 50× compute difference either way.

Untested. Stated here as a hypothesis so it is falsifiable, not as a finding.

## Protocol requirements (LeRobot docs)

- `MUJOCO_GL=egl`
- hard resets — soft resets are not bit-identical and change camera observations
- `--env.init_states=true` and a fixed `--seed` for comparability across runs
- `--env.control_mode` must match the checkpoint's action parameterisation
- pin `--dataset.revision` when reporting

## Gate — DECIDED 2026-09-11 (D2)

**10 episodes/task, 400 episodes total**, matching LeRobot's stated protocol.

The proposal's ±5 pp tolerance **does not apply at this n** — the 95% Wilson
interval at n=100/suite is ~±6–9 pp, wider than the tolerance itself. Replaced
by a **CI-overlap test** (`PLAN.md` §5.3):

- published **inside** our CI ⇒ *consistent with published* (not "reproduced" —
  we cannot distinguish). Proceed, and report the interval, never a point.
- published **outside** ⇒ genuine non-reproduction; a finding. Escalate that
  suite to 50 eps/task before reporting.
- CI contains **both** published (~87) and the failed-repro value (~73) ⇒ the
  measurement is too weak to say anything. Escalate. Report neither.

Given two open non-reproductions upstream, "cannot reproduce" is a *reportable
finding*, not a blocker.

**The first full run is an ORIENTING run, not a reported result.** Label it so
in the output so it cannot be quoted as a baseline later.
