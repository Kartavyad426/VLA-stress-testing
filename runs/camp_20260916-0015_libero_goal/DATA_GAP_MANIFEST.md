# Data Gap Manifest

Generated 2026-09-16 · schema v1.0

## DGM-001 — ambiguous [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_goal/task0, libero_goal/task1, libero_goal/task2, libero_goal/task3, libero_goal/task4, libero_goal/task5
- **Symptom:** 104 failures terminating as ; mean final error 0.0 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +100 pp (0% → 100%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 260 episodes · nominal 23% · 104 failures in cluster
- **Coverage required:** Targeted coverage of the failing region.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-002 — planning [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_goal/task6, libero_goal/task7
- **Symptom:** 27 failures terminating as never_reached; mean final error 19.5 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +100 pp (0% → 100%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 260 episodes · nominal 89% · 27 failures in cluster
- **Coverage required:** Longer-horizon and subgoal-rich demonstrations covering the phase where the policy terminates early.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-003 — ambiguous [NOT RANKABLE - PREVALENCE NOT ESTIMATED]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_goal/task0, libero_goal/task1, libero_goal/task2, libero_goal/task3, libero_goal/task4, libero_goal/task5
- **Symptom:** 15 failures terminating as ; mean final error 0.0 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +100 pp (0% → 100%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 260 episodes · nominal 23% · 15 failures in cluster
- **Coverage required:** Targeted coverage of the failing region.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM
