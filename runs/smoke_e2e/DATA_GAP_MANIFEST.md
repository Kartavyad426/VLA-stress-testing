# Data Gap Manifest

Generated 2026-09-15 · schema v1.0

## DGM-001 — ambiguous [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_spatial/task0, libero_spatial/task1
- **Symptom:** 3 failures terminating as failed_grasp_no_retry/retry_loop; mean final error 0.0 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +50 pp (0% → 50%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 20 episodes · nominal 50% · 3 failures in cluster
- **Coverage required:** Targeted coverage of the failing region.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-002 — ambiguous [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_spatial/task0, libero_spatial/task1
- **Symptom:** 2 failures terminating as failed_grasp_no_retry/retry_loop; mean final error 0.0 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +50 pp (0% → 50%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 20 episodes · nominal 50% · 2 failures in cluster
- **Coverage required:** Targeted coverage of the failing region.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-003 — planning [COMPUTED AT DELIVERY = CONDITIONAL X PREVALENCE X COST]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** libero_spatial/task0, libero_spatial/task1
- **Symptom:** 2 failures terminating as never_reached; mean final error 0.0 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +50 pp (0% → 50%)
- **Boundary:** not crossed in the swept range
- **Evidence:** 20 episodes · nominal 80% · 2 failures in cluster
- **Coverage required:** Longer-horizon and subgoal-rich demonstrations covering the phase where the policy terminates early.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM
