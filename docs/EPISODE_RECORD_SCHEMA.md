# Episode record schema — what R-039 / R-040 consume, and who fills each field

*Written 2026-09-23. Agreed between the `fable` and `primary` sessions on
2026-09-23; this file is the first place it is written down. `primary` owns
`experiments/vlm_label.py`; the simulator-derived fields have no owner yet.*

## Why one record

Localisation (R-039) says **which pathway** carried a failure. It does not
produce demonstrations. Every retraining arm in R-040 still needs successful
demonstrations of the failed situation, and arm M needs to pick them by
**the axis the failure sits on**, checked against the training corpus. That
check is a count, so the fields it counts on must exist with the same keys
over failures and over training demos. One record per episode, one schema for
both sets, is the whole point.

## Two sources, one record

| field | type | source | why |
|---|---|---|---|
| `episode_id`, `task_id`, `variant`, `level`, `seed` | ids | trace | join key |
| `scene` | short name | variant name | e.g. `next-to-ramekin` |
| `observed_condition.category` | enum | LIBERO-Plus catalogue for failures; `nominal` for training demos | what LOOKS perturbed; **never named `mechanism` or `cause`** (Theorem 2: moves ≠ repairs) |
| `observed_condition.view` | (yaw, pitch, scale, endrot, endvert) or null | variant name / knobs | family B, viewpoint |
| `observed_condition.light` | light id or intensity | variant name / knobs | family B, illumination |
| `observed_condition.noise` | severity 0–50 | variant name | family B, sensor |
| `observed_condition.joint_radius_rad` | float | `‖init_qpos − canonical‖` from sim | family A1, start pose |
| `target_object_class` | string | BDDL `obj_of_interest` | family A2 |
| `target_pose_bin` | (x-bin, y-bin) in agent view | sim, `_gt_object_pos` at step 0 | family A1 placement |
| `n_similar_candidates` | int | BDDL: same-class objects in scene | the R-038 covariate |
| `referring_expression_form` | enum: `preposition-relative` / `name-only` / `ordinal` / `colour` | instruction string, rule-based | family A3 |
| `clutter_count` | int | BDDL object count | family B |
| `outcome.success`, `outcome.termination` | bool, enum | simulator | ground truth |
| `anchor.closest_approach_step` | int | trace, `_gt_eef_to_object` to the target **bowl** | R-038's discriminator; primary anchor |
| `anchor.closest_approach_m` | float | same | |
| `anchor.closest_approach_forward` | int | logged forward env-steps (`forward_env_steps`), **not** `step // 16` | R-039 splices per forward |
| `anchor.grasp_step` | int or null | first step any object rises > 1 cm, from `_gt_scene_object_pos` | secondary anchor. Not "gripper close + contact": `_gt_n_contacts` counts every contact in the scene, and the gripper closes on air repeatedly in failures |
| `handled.objects` | list of {object, moved_m, lifted_m, is_target} | `_gt_scene_object_pos` (every free-joint body; added 2026-09-23) | **which object the arm actually handled.** `_gt_object_pos` holds only BDDL task objects, so a wrong-object grasp is invisible in it: 5115b970e766 reads "nothing moved" while the ramekin was lifted 6 cm |
| `handled.wrong_object` | bool | `handled.objects`: a non-target moved > 1 cm | the ground truth `attempt.grasped_object` is scored against |
| `attempt.completed`, `attempt.grasped_object`, `attempt.approached_target`, `attempt.summary` | VLM | `vlm_label.py`, blind to outcome and perturbation | how the attempt unfolded; the ONLY VLM fields |
| `attempt.model_id`, `attempt.prompt_sha`, `attempt.agreement_with_sim` | provenance | labeller | a label is a measurement; calibrate before counting. First datapoint (5115b970e766): Nemotron-Omni and Qwen3-VL-8B both detected the grasp but named the wrong object (target bowl, not the ramekin) and wrongly said the arm approached the target |
| `pathway` (R-039 only) | `{T, I, S}` transfer at f0 and at anchor, dominant arm | `runs/r039*/analysis.json` | the localisation result, joined by `episode_id` |

Rules agreed:

1. **Anything the simulator or BDDL holds exactly is read from there, never
   estimated by a VLM.** The VLM labels how the attempt unfolded, nothing
   else. (primary's correction; withdrawn: my request for VLM labels on the
   training set.)
2. **Same keys over failures and over training demos.** Corpus counts for
   family claims are computed on the training-demo records; the training
   demos come from the LIBERO hdf5 with full sim states, replayed.
3. **`observed_condition` is a description, `pathway` is a measurement.**
   R-038 is the standing example of the two disagreeing.
4. **Anchors are recorded per episode**, first frame and closest-approach
   frame separately, because R-039 distinguishes "wrong at forward 0" from
   "drifted into it".

## What R-040's arms read

- **D** (direct corrective): `episode_id` of each failure → collect a
  successful demo of that exact `task_id`/`variant`/`seed`.
- **M** (mechanism-targeted): `pathway.dominant` × `observed_condition` →
  the axis; then `target_pose_bin` / `view` / `light` counts over the
  training-demo records say how under-sampled that axis is and where to
  re-render or collect.
- **O** (OOD-ranked): none of the above; the proprio detector's score.
- **R** (nominal): none.

After R-039, the expected `pathway.dominant` for every LIBERO-Plus category
is the image tokens, including robot initial state; the axis still comes
from `observed_condition`, the pathway says what kind of data reaches it
(rendered, not proprio-only).
