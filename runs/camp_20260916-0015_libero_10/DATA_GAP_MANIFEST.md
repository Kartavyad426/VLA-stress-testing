# Data Gap Manifest

Generated 2026-09-16 · schema v1.0

## DGM-001 — visual_grounding [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_10/task1, libero_10/task2, libero_10/task3, libero_10/task5, libero_10/task6
- **Symptom:** 39 failures terminating as failed_grasp_no_retry/retry_loop/unknown; mean final error 27.4 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +40 pp (40% → 80%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 185 episodes · nominal 68% · 39 failures in cluster
- **Coverage required:** Demonstrations spanning the measured camera_yaw_deg band in even steps, across multiple object layouts and both target-object classes.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-002 — planning [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_10/task1, libero_10/task2, libero_10/task3, libero_10/task5, libero_10/task6
- **Symptom:** 19 failures terminating as never_reached; mean final error 28.5 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +40 pp (40% → 80%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 185 episodes · nominal 85% · 19 failures in cluster
- **Coverage required:** Longer-horizon and subgoal-rich demonstrations covering the phase where the policy terminates early.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-003 — manipulation [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_10/task3, libero_10/task6
- **Symptom:** 19 failures terminating as failed_grasp_no_retry/retry_loop; mean final error 5.5 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +40 pp (40% → 80%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 185 episodes · nominal 86% · 19 failures in cluster
- **Coverage required:** Trajectory variants around contact and alignment across the failing camera_yaw_deg band; include successful corrections.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM
