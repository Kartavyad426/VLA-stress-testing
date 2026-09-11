# Data Gap Manifest

Generated 2026-09-11 · schema v1.0

## DGM-001 — visual_grounding [HIGH]

- **Evidence strength:** causal (counterfactual-confirmed)
- **Tasks:** pick_bowl
- **Symptom:** 80 failures terminating as failed_grasp_no_retry/retry_loop; mean final error 18.4 cm
- **Trigger:** camera_yaw_deg — counterfactual probe: reverting this knob alone recovers +95 pp (5% → 100%)
- **Boundary:** camera_yaw_deg between 3 and 6 (success 100% → 0%; definition: success < 0.5 x nominal (100.0%))
- **Evidence:** 180 episodes · nominal 100% · 80 failures in cluster
- **Supply-side corroboration:** training demos cluster within +/-8 deg yaw (Phase 0 — PLACEHOLDER, not yet measured)
- **Coverage required:** Demonstrations spanning the measured camera_yaw_deg band in even steps, across multiple object layouts and both target-object classes.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM

## DGM-002 — manipulation [MEDIUM]

- **Evidence strength:** correlational
- **Tasks:** pick_bowl
- **Symptom:** 40 failures terminating as failed_grasp_no_retry; mean final error 5.9 cm
- **Trigger:** camera_yaw_deg — co-occurs with failure; NOT counterfactual-confirmed
- **Boundary:** not crossed in the swept range
- **Evidence:** 180 episodes · nominal 100% · 40 failures in cluster
- **Coverage required:** Trajectory variants around contact and alignment across the failing camera_yaw_deg band; include successful corrections.
- **Validation:** LoRA fine-tune at three escalating data budgets on the coverage above; re-run the frozen regression set; report the data-response curve. Neutral and negative results are reportable.
- **Validation result:** `null` — UNTESTED CLAIM
