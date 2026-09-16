# Data Gap Manifest

Generated 2026-09-16 · schema v1.0

## DGM-001 — visual_grounding [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_object/task0, libero_object/task1, libero_object/task2, libero_object/task3, libero_object/task4, libero_object/task5, libero_object/task6, libero_object/task7
- **Symptom:** 143 failures terminating as retry_loop; mean final error 40.9 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +60 pp (0% → 60%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 260 episodes · nominal 37% · 143 failures in cluster
- **Coverage required:** Demonstrations spanning the measured camera_yaw_deg band in even steps, across multiple object layouts and both target-object classes.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-002 — spatial_reasoning [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_object/task1, libero_object/task5
- **Symptom:** 12 failures terminating as retry_loop; mean final error 36.0 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +60 pp (0% → 60%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 260 episodes · nominal 94% · 12 failures in cluster
- **Coverage required:** Demonstrations varying target position against distractor count and placement; include hard negatives where the nearest object is not the target.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-003 — visual_grounding [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_object/task0, libero_object/task1, libero_object/task3, libero_object/task4, libero_object/task5, libero_object/task6
- **Symptom:** 6 failures terminating as retry_loop; mean final error 34.4 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +60 pp (0% → 60%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 260 episodes · nominal 37% · 6 failures in cluster
- **Coverage required:** Demonstrations spanning the measured unknown band in even steps, across multiple object layouts and both target-object classes.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM
