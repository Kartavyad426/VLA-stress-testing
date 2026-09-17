# Data Gap Manifest

Generated 2026-09-17 · schema v1.0

## DGM-001 — visual_grounding [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** correlational
- **Tasks:** libero_goal/task1, libero_goal/task2, libero_goal/task3, libero_goal/task4, libero_goal/task6
- **Symptom:** 26 failures terminating as failed_grasp_no_retry/retry_loop; mean final error 30.2 cm
- **Trigger:** camera_yaw_deg — co-occurs with failure; NOT counterfactual-confirmed
- **Boundary:** not crossed in the swept range
- **Evidence:** 265 episodes · nominal 86% · 26 failures in cluster
- **Coverage required:** Demonstrations spanning the measured camera_yaw_deg band in even steps, across multiple object layouts and both target-object classes.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-002 — visual_grounding [NOT RANKABLE - PREVALENCE NOT ESTIMATED]

- **Evidence strength:** correlational
- **Tasks:** libero_goal/task0, libero_goal/task1, libero_goal/task2, libero_goal/task3, libero_goal/task4, libero_goal/task6
- **Symptom:** 11 failures terminating as failed_grasp_no_retry/retry_loop; mean final error 30.7 cm
- **Trigger:** unknown — co-occurs with failure; NOT counterfactual-confirmed
- **Boundary:** not crossed in the swept range
- **Evidence:** 265 episodes · nominal 86% · 11 failures in cluster
- **Coverage required:** Demonstrations spanning the measured unknown band in even steps, across multiple object layouts and both target-object classes.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-003 — planning [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** correlational
- **Tasks:** libero_goal/task0, libero_goal/task3
- **Symptom:** 11 failures terminating as never_reached; mean final error 26.7 cm
- **Trigger:** camera_yaw_deg — co-occurs with failure; NOT counterfactual-confirmed
- **Boundary:** not crossed in the swept range
- **Evidence:** 265 episodes · nominal 92% · 11 failures in cluster
- **Coverage required:** Longer-horizon and subgoal-rich demonstrations covering the phase where the policy terminates early.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM
