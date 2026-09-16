# Data Gap Manifest

Generated 2026-09-16 · schema v1.0

## DGM-001 — visual_grounding [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_spatial/task0, libero_spatial/task1, libero_spatial/task2, libero_spatial/task3, libero_spatial/task5, libero_spatial/task6
- **Symptom:** 42 failures terminating as failed_grasp_no_retry/retry_loop; mean final error 16.0 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +60 pp (0% → 60%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 260 episodes · nominal 76% · 42 failures in cluster
- **Coverage required:** Demonstrations spanning the measured camera_yaw_deg band in even steps, across multiple object layouts and both target-object classes.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-002 — manipulation [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_spatial/task0, libero_spatial/task1, libero_spatial/task2, libero_spatial/task3, libero_spatial/task5, libero_spatial/task6, libero_spatial/task7
- **Symptom:** 37 failures terminating as failed_grasp_no_retry/retry_loop/unknown; mean final error 7.5 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +60 pp (0% → 60%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 260 episodes · nominal 83% · 37 failures in cluster
- **Coverage required:** Trajectory variants around contact and alignment across the failing camera_yaw_deg band; include successful corrections.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-003 — planning [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_spatial/task0, libero_spatial/task1, libero_spatial/task3, libero_spatial/task4, libero_spatial/task7
- **Symptom:** 34 failures terminating as never_reached; mean final error 14.4 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +60 pp (0% → 60%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 260 episodes · nominal 85% · 34 failures in cluster
- **Coverage required:** Longer-horizon and subgoal-rich demonstrations covering the phase where the policy terminates early.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM
