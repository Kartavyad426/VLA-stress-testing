**VLA Stress-Test**  
 **{with Failure Mining}**

From simulation failures to targeted multimodal data requirements

| Project mode | Focused public-data proof of concept with a clear scale-out path |
| :---- | :---- |
| Data / models | Public benchmarks, public/open models, synthetic simulation data only |
| Primary outcome | A reproducible failure-mining pipeline plus a Data Gap Manifest |

**Core thesis**

| Systematically stress a VLA policy, identify where and why it fails, cluster those failures, convert them into precise requirements for the next data-collection or training cycle, and validate at least one of those requirements by retraining and re-measuring. |
| :---: |

**Why this matters:** robotics teams do not just need another benchmark score. They need to know the operating envelope of a policy, the failure modes that matter, and what data will close those gaps.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

# **1\. Executive Summary**

Vision-Language-Action (VLA) models are moving robotics from task-specific policies toward generalist systems that interpret language, perceive scenes and generate actions. The deployment challenge is no longer only whether a policy can complete a benchmark task; it is whether that policy remains reliable when viewpoints, lighting, object positions, distractors, language phrasing and execution conditions change.

This project builds a closed-loop evaluation framework that subjects an open VLA policy to controlled perturbations in simulation, records the complete multimodal rollout, identifies the point and type of failure, clusters recurring failure patterns, and translates those patterns into actionable data requirements.

The result is not another leaderboard. It is a repeatable method for turning model behaviour into a data strategy: what failed, under what conditions, how often, why it matters, and what additional demonstrations or examples should be collected next.

## **Definition of success**

| 1 Stress Run controlled perturbations against a fixed policy. | 2 Mine Detect and group recurring failure patterns. | 3 Explain Identify the stage, trigger and likely failure class. | 4 Recommend Produce a quantified Data Gap Manifest for the next training cycle. |
| :---: | :---: | :---: | :---: |

# **2\. Problem Statement**

**Aggregate task success is an incomplete measure of a robot policy.** A model may score well on a standard task while having a narrow operating envelope: a small camera shift, mild occlusion, a paraphrased instruction or a different object arrangement may cause the policy to fail abruptly. In physical systems, those hidden boundaries matter more than a single average score.

Three gaps motivate the project:

·       **Coverage gap:** Conventional evaluation often samples too few environment variations to reveal where robustness breaks.

·       **Diagnosis gap:** A failed rollout usually ends as success/failure telemetry. Teams still need to understand whether the root cause was perception, language grounding, spatial reasoning, planning, manipulation or recovery.

·       **Data-action gap:** Even when a weakness is known, the result is rarely translated into a concrete recommendation for what additional data should be collected, generated, curated or reviewed.

**The project closes all three gaps in one loop:** systematic simulation-based stress testing → trace-level failure mining → data-gap recommendation → re-evaluation.

# **3\. Why This Is an Industry-Relevant Problem**

**Simulation is becoming an evaluation system, not just a training tool.** Modern robotics frameworks support repeatable, parameterized environments and large-scale parallel policy evaluation. This makes it practical to map robustness across thousands of controlled variations rather than relying on a handful of physical trials. \[1\]

**Generalist policies increase the need for operating-envelope testing.** As one policy is expected to generalize across objects, scenes, embodiments and instructions, failures become distributional: a model can succeed on the task yet fail under a small shift in the observation or instruction. Evaluation therefore must measure robustness boundaries, not only nominal success. Published robustness analysis of current VLA models on this benchmark family reports success falling from roughly 95% to under 30% under modest camera-viewpoint and robot-initial-state changes, so the effect is large and already measurable. \[11\]

**The valuable artifact is increasingly the trace.** A rollout contains video, language, robot state, actions, environment parameters and outcome. Treating these as a governed, queryable evaluation corpus makes it possible to compare model versions, reproduce failures, create golden regression sets and retain lineage from failure to remediation. \[2\]\[3\]

**Continuous improvement needs a bridge from model failure to data operations.** Cloud reference architectures for Physical AI increasingly connect simulation, sensor data, retraining, monitoring and redeployment as one continuous cycle. The missing intelligence layer is deciding which failures deserve attention and what data should be added next. \[4\]

# **4\. What the Project Will Build**

The project should build the smallest credible end-to-end system that proves the thesis. The recommended fast path uses OpenVLA-OFT in its native LIBERO evaluation stack as the first policy, with LIBERO-plus providing the robustness environment through a thin benchmark adapter that preserves the policy inference path. Hugging Face LeRobot is used where it provides first-class policy and environment support, particularly for NVIDIA Isaac GR00T N1.7 and LIBERO/LIBERO-plus. LIBERO-plus already supplies the perturbation machinery \- 10,030 task instances across seven perturbation factors and 21 sub-dimensions, stratified into difficulty levels L1-L5 \- so stress generation is an integration task rather than a contribution. \[11\] The contribution is what happens after the failure: stage localization, counterfactual attribution, data-gap translation and validated remediation. Higher-fidelity or GPU-parallel simulation (Isaac Sim, Isaac Lab and Isaac Lab-Arena) is outside the initial project scope and is retained in section 7.2 as the scale-out path. All policy and simulator adapters should emit the same canonical rollout schema so the failure-mining layer remains model- and simulator-agnostic.

| 1\. VLA Policy \+ Public Task Suite Run a fixed policy on a reproducible set of manipulation tasks. |
| :---- |
| **2\. Controlled Stress Generator** Vary viewpoint, lighting, object pose, clutter, occlusion, initial state and instruction phrasing. |
| **3\. Multimodal Rollout Capture** Record observations, language, robot state, action trajectory, environment parameters, timing and outcome. |
| 4\. Failure Detection & Stage Localization Identify the first meaningful divergence and the task phase in which the rollout became unrecoverable. Build explicit phase detectors from simulator state and contact signals \- for example: approach (end-effector-to-target distance), pre-grasp alignment, grasp (gripper closure/contact/object lift), transport, placement or target relation achieved, release, and final LIBERO goal satisfaction. Use BDDL predicates where they represent genuine semantic milestones, especially final goal relations, but do not assume every task exposes reach/grasp/transport/place predicates. Use trajectory distance only as a secondary within-phase signal because successful manipulation can follow multiple valid trajectories. |
| **5\. Failure Classification & Clustering** Group failures by behaviour, trigger, visual/semantic similarity and trajectory pattern, then attribute cause with counterfactual probes \- re-run the same seed reverting one perturbation dimension at a time \- rather than inferring cause from the trajectory alone. |
| **6\. Data Gap Manifest** Translate dominant failure clusters into quantified requirements for additional demonstrations, edge cases or synthetic variants. |
| **7\. Regression Set** Promote representative failures into a reusable evaluation set for future model versions. |

## **Recommended initial perturbation set**

| Dimension | Examples | What it tests | Output |
| :---- | :---- | :---- | :---- |
| Viewpoint | Camera yaw/pitch/offset | Visual grounding & spatial robustness | Failure boundary by angle/offset |
| Scene | Object pose, clutter, distractors | Generalization & object selection | Success by scene complexity |
| Visibility | Partial occlusion, illumination | Perception robustness | Failure rate by visibility level |
| Language | Paraphrase, spatial wording | Language grounding | Sensitivity to instruction form |
| Execution | Initial state, slight perturbation | Control & recovery | Recovery rate / repeat-failure rate |

**Prioritisation.** Published results on this benchmark family show camera viewpoint and robot initial state produce the largest degradation, while language perturbations produce comparatively little change. Run viewpoint, initial state and object layout first; treat the language axis as a stretch goal. \[11\]

# **5\. Failure Mining: The Differentiating Layer**

1\.         Where did the trajectory first deviate from a successful execution?

2\.         What kind of failure was it?

3\.         Which environment or instruction change is most correlated with that failure?

4\.         What additional data would increase coverage of the weak region?

5\.         Did that data close the gap when the policy was retrained and the same conditions were re-run?&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

## **Initial failure taxonomy**

| Failure family | Typical symptom | Likely remediation signal |
| :---- | :---- | :---- |
| Visual grounding | Approaches wrong object or loses target under occlusion | More viewpoint / occlusion / distractor coverage |
| Language grounding | Different paraphrase causes different or incorrect behaviour | Instruction diversity and semantic contrast cases |
| Spatial reasoning | Correct object, wrong spatial relation or destination | Spatially varied demonstrations and harder negatives |
| Planning | Correct start, incorrect sequence or premature termination | Longer-horizon or subgoal-rich demonstrations |
| Manipulation / control | Reaches target but grasp, placement or alignment fails | Trajectory variants around contact and alignment |
| Recovery | First attempt fails and policy loops or gives up | Failed-attempt \+ recovery demonstrations |
| Distribution shift | Small visual/environment change causes abrupt degradation | Targeted domain variation around the failure boundary |

## **The key output: Data Gap Manifest**

The final artifact should convert model behaviour into a prioritized data plan. Every row must carry the evidence behind it \- rollouts run, observed success rate with confidence interval, and the perturbation level at which the boundary was crossed. The recommendation should specify the weak operating region and the coverage required to remediate it, rather than inventing a sample count. Where sample volumes are proposed, derive them through staged remediation experiments (for example, increasing targeted data budgets) and report the resulting data-response curve. At least one manifest row must be validated by fine-tuning and re-evaluation. Example:

| Task | Observed failure | Trigger | Severity | Remediation coverage / validation plan |
| :---- | :---- | :---- | :---- | :---- |
| Pick object | Visual grounding | Camera yaw \+15° | High | Cover the measured viewpoint boundary (for example \+10° to \+20°) across multiple object layouts; validate increasing targeted sample budgets. |
| Place object | Grasp / control | Initial-state offset \+ occlusion | High | Cover end-effector start offsets, approach directions and occlusion bands around the failing grasp regime; include successful correction trajectories. |
| Pick target object | Object selection | Clutter \+ distractors | Medium | Cover distractor count, target position and appearance combinations around the measured selection boundary; include hard negatives. |
| Pick after miss | Recovery | Failed first grasp | High | Add failed-attempt \+ successful-recovery trajectories with varied miss direction and contact state; measure whether recovery success improves. |

# **6\. System Architecture and Data Flow**

The architecture is intentionally designed so that evaluation can start with a lightweight public benchmark and later scale to GPU-parallel simulation without changing the failure-intelligence layer. The fast path uses the native OpenVLA-OFT evaluation stack with LIBERO/LIBERO-plus; GR00T N1.7 is evaluated through LeRobot as the second-policy path. The scale path uses Isaac Sim, Isaac Lab and Isaac Lab-Arena for higher-fidelity and larger-volume qualification. All paths produce the same multimodal rollout contract.

| SIMULATE Parameterized environments Parallel rollouts | CAPTURE Video \+ language state \+ actions | GOVERN Versioned traces metadata \+ lineage | MINE Failure scoring clustering \+ review | IMPROVE Data gap manifest regression set |
| :---: | :---: | :---: | :---: | :---: |

Design principle: simulation produces evidence; a versioned data layer preserves that evidence with lineage; analytics turns it into failure intelligence; expert review resolves ambiguous cases; and the resulting gaps become inputs to the next model/data cycle. The technology stack can evolve independently at each layer as long as the rollout and evaluation contracts remain stable.

# **7\. Reference Technology Stack**

The stack is deliberately split into two layers. The first is a fast, reproducible robustness loop using public benchmarks and pre-trained policies. The second is a scale-out path for richer simulation, larger rollout volumes, additional embodiments and enterprise-grade data lineage. Both paths emit the same rollout contract so the failure-mining and data-gap layers do not need to be rewritten when the simulator or policy changes.

## **7.1 Rapid robustness evaluation path**

| Layer | Recommended technology | Role in the project | Implementation notes |
| :---- | :---- | :---- | :---- |
| Evaluation adapters | OpenVLA-OFT native evaluator; Hugging Face LeRobot for supported policies | Run each policy in its supported stack while emitting one canonical rollout and evaluation contract. | Do not force OpenVLA-OFT through LeRobot: current LeRobot does not provide first-class OpenVLA-OFT support. Use separate pinned environments/containers for vanilla LIBERO and LIBERO-plus because LIBERO-plus replaces the vanilla libero package. \[5\]\[6\] |
| Base simulation benchmark | LIBERO | Public manipulation tasks covering spatial, object, goal and long-horizon behaviours. | Use the standard suites for nominal baseline reproduction. OpenVLA-OFT should use its canonical native LIBERO evaluation path. LIBERO runs on MuJoCo. \[6\]\[9\] |
| Robustness benchmark | LIBERO-plus | Systematic stress testing across camera viewpoint, object layout, robot initial state, language, lighting, textures and sensor noise. | Use LIBERO-plus as the primary perturbation generator rather than hand-coding every robustness variant. \[5\]\[11\] Keep it in a separate environment/container from vanilla LIBERO, and preserve identical policy preprocessing/action conventions across the two environments. |
| Reference policy | OpenVLA-OFT | Strong open VLA baseline with published LIBERO checkpoints and evaluation scripts. | Start here to validate the end-to-end stress-test and trace pipeline quickly using the canonical moojink/openvla-oft repository. The canonical evaluator runs 10 tasks x 50 episodes (500 trials) per suite by default; the published setup used Python 3.10.14, PyTorch 2.2.0 and the project's custom Transformers v4.40.1 fork, averaged over three random seeds. Pin the suite-specific action un-normalization configuration and GPU/runtime because environment mismatches can manufacture apparent model failures. \[6\] |
| Second policy | NVIDIA Isaac GR00T N1.7 \- LIBERO checkpoint | Cross-model comparison and a direct path into a broader generalist robotics stack. | Use the official NVIDIA LIBERO checkpoint through the current LeRobot GR00T integration; match the policy action representation/control mode to the environment. GR00T N1.7 is now a General Availability release with support and stability guarantees. Pin both the repository commit and checkpoint revision. \[7\] |
| Rollout format | LeRobot / JSONL \+ Parquet \+ MP4 | Store instruction, images/video, robot state, actions, environment parameters, model/checkpoint, timing and outcome. | Keep one canonical schema across all simulators and policy families. |

## **7.2 Scale-out simulation and qualification path (future direction)**

| Layer | Recommended technology | Role in the project | Implementation notes |
| :---- | :---- | :---- | :---- |
| Simulation runtime | NVIDIA Isaac Sim | Higher-fidelity physics, sensor simulation and domain randomization for selected high-value failure families. | Use after the LIBERO loop is working; do not make simulator construction the first milestone. |
| Robot learning framework | NVIDIA Isaac Lab | Task/environment authoring, robot learning workflows and scalable simulation built on Isaac Sim. | Provides the foundation for moving beyond a single benchmark or embodiment. \[1\] |
| Large-scale evaluation | NVIDIA Isaac Lab-Arena | GPU-parallel policy evaluation across diversified tasks, objects, environments and embodiments. | Use for large stress matrices and regression suites once the failure taxonomy is stable. \[1\] |
| Generalist VLA | NVIDIA Isaac GR00T N1.7 | Primary model family for the scale-out path; supports inference, fine-tuning and simulation/real evaluation workflows. | Keep model post-training outside the core evaluation project; use fixed checkpoints when comparing robustness. \[7\] |
| Video/text feature layer | NVIDIA Cosmos Embed1 (optional) | Joint video-text embeddings for failure retrieval, semantic similarity, clustering and representative-example selection. | Use failure-centered short clips rather than entire rollouts because Cosmos Embed1 is optimized for short-form video. Combine the clip embedding with task instruction, perturbation metadata and failure-stage labels. \[8\] |
| Compute orchestration | Containerized GPU jobs; AWS EC2 / EKS / Batch when scale is needed | Parallelize simulation and policy inference without changing the evaluation contract. | Local GPU is sufficient for the initial loop; scale selected experiments rather than everything. \[4\] |
| Artifact storage | Amazon S3 or equivalent object storage | Persist rollout videos, simulator state, environment configs, logs and model artifacts. | Use immutable experiment prefixes and content/version metadata. |
| Governed data & analytics | Databricks Unity Catalog \+ Delta tables \+ MLflow Tracking / Model Registry | Govern unstructured video and structured rollout metadata; retain lineage; compare model, environment, data and perturbation versions. | Store/query rollout metadata in Delta and link to governed video artifacts. Use MLflow to log runs, model/checkpoint versions, parameters, metrics and artifacts; use Models in Unity Catalog where model lifecycle governance is needed. \[2\]\[3\] |
| Human adjudication | Ango Hub | Review ambiguous root causes, confirm taxonomy labels and promote gold regression cases. | Review a stratified sample of high-value or ambiguous failures rather than every rollout. Double-label a subset and report inter-annotator agreement on the failure taxonomy; preserve adjudicated cases as gold regression examples. |

## **7.3 Recommended implementation order**

1\. Baseline: run the canonical OpenVLA-OFT checkpoints on all 10 LIBERO-Spatial tasks and all 10 LIBERO-Object tasks, using 50 episodes per task and the canonical evaluation configuration. Acceptance gate: per-suite success should reproduce the canonical published result within \+/-5 percentage points before any perturbation work begins. If it does not, repeat with additional seeds and debug the environment before proceeding. Record repository SHA, checkpoint revision, GPU type, Python/PyTorch/Transformers versions and simulator dependencies.

2\. Stress: switch to LIBERO-plus and stress a representative 10-task subset (for example five Spatial and five Object tasks) across camera viewpoint, robot initial state and object layout at four difficulty levels. Use approximately 20 episodes per cell for screening, then increase to approximately 50 episodes in cells around the apparent failure boundary. Report success rates with 95% confidence intervals rather than point estimates.

3\. Cross-model (future direction): run the same selected tasks and perturbations using the official GR00T N1.7 LIBERO checkpoint through LeRobot, keeping task IDs, seeds and environment conditions aligned where possible. The policies differ in size, backbone, action representation and control mode, so this produces two robustness profiles rather than a universal model ranking.

4\. Mine: localize failure phase using simulator-state phase detectors, extract trajectory and visual features, classify failures and cluster recurring patterns. Attribute the likely trigger using counterfactual probes \- re-run the same seed while reverting one perturbation dimension at a time. Use Cosmos Embed1 only if short failure-centered video/text embeddings materially improve clustering or retrieval.

5\. Validate: fine-tune on targeted remediation data for the single highest-severity Data Gap Manifest row, re-run the frozen regression set and report the change in the failure boundary. At least one remediation budget is required to close the loop; where compute permits, test increasing data budgets to estimate a data-response curve. The result may be positive, neutral or negative \- all three are informative.

6\. Operationalize: persist rollout videos and metadata in versioned object storage, retain experiment lineage and freeze representative failures as a reusable regression set. A local object store plus Parquet/MP4 and MLflow is sufficient for the initial implementation; the canonical schema should map cleanly to S3, Delta/Unity Catalog and governed model/experiment lineage when the workflow scales.

# **8\. Strategic Relevance to iMerit \+ EXL**

This project sits at the intersection of model evaluation, multimodal data operations and enterprise AI infrastructure. That intersection is strategically more important than the benchmark itself.

**For iMerit:** Moves the conversation from annotation volume to model-aware data intelligence. iMerit can contribute robotics domain expertise, multimodal trace review, failure taxonomies, human adjudication and targeted curation around high-value edge cases.

**For EXL:** Creates a repeatable Physical AI evaluation pattern that can be operationalized with governed data, analytics, scalable compute, experiment lineage and enterprise controls rather than remaining a research-only demo.

**For the combined proposition:** Connects two normally separate conversations: “How good is the model?” and “What data should we invest in next?” The combined service can measure the first and systematically answer the second. \[10\]

## **Potential market-facing proposition**

| “We identify where a Physical AI model fails, quantify the operating conditions that trigger those failures, and turn them into a prioritized data and evaluation plan for the next model iteration.” |
| :---: |

This can evolve into a service layer spanning:

·       Simulation-based policy qualification and regression testing.

·       Multimodal rollout analysis and failure taxonomy creation.

·       Human expert review of ambiguous or safety-relevant failures.

·       Data-gap analysis and targeted collection / synthetic-data recommendations.

·       Versioned evaluation sets that follow a model across releases.

·       Quality, robustness, latency and cost dashboards for policy comparison.

# **9\. Recommended Initial Project Scope**

| Phase | Work | Deliverable |
| :---- | :---- | :---- |
| 1\. Baseline | Run OpenVLA-OFT on all 10 LIBERO-Spatial and all 10 LIBERO-Object tasks, 50 episodes per task. Reproduce canonical per-suite success within \+/-5 percentage points before proceeding. | Baseline success with confidence intervals, complete rollout traces, pinned environment metadata and a pass/fail result on the reproduction gate |
| 2\. Stress matrix | Stress 10 representative tasks (five Spatial, five Object) across camera viewpoint, robot initial state and object layout at four levels. Use \~20 episodes/cell for screening and \~50 around the failure boundary; state the resulting GPU-hours. | Robustness curves with 95% confidence intervals and one or more measured failure boundaries |
| 3\. Failure mining | Localize failure phase, classify and cluster failed rollouts, attribute likely triggers with counterfactual probes, and double-label a human-reviewed subset to measure taxonomy agreement. | Failure taxonomy \+ cluster report |
| 4\. Data translation | Convert dominant clusters into evidence-backed remediation coverage requirements; where sample counts are proposed, justify them through staged data budgets rather than fixed guesses. | Data Gap Manifest |
| 5\. Regression | Freeze representative failures as a future test set, fine-tune on targeted remediation data for the top manifest row, and re-run the regression set. | Reusable regression suite, before/after robustness measurement, and demo |

## **Required deliverables**

·       Reproducible code and pinned environment.

·       Baseline and stress-test results with complete multimodal traces.

·       Failure taxonomy with representative examples.

·       Failure-cluster visualization and robustness curves.

·       Data Gap Manifest with prioritized data recommendations.

·       A compact demo showing one nominal success, one controlled failure, its diagnosis and the resulting data recommendation.

·       Concise technical/executive readout explaining findings, limits and next steps.

## **Success criteria**

·       At least three failure modes are reproduced with confidence intervals, and at least one is characterized at task/stage specificity beyond the aggregate benchmark result with an actionable remediation hypothesis.

·       At least one failure boundary is measured under an explicit definition \- for example, the perturbation magnitude at which mean success falls below half the nominal rate \- and reported with a confidence interval.

·       Failure clusters can be traced back to concrete environment/instruction conditions. Attribution rests on counterfactual probes, not clustering alone; inter-annotator agreement is reported for a double-labelled human-review subset.

·       The Data Gap Manifest is specific enough that a separate team could act on it without re-reading every rollout.

·       Representative failures are promoted into a reproducible regression set.

# **10\. Natural Follow-On**

The natural follow-on is an Intelligent Robotics Data Curator. This project establishes the demand signal: which capabilities are weak, where the operating envelope breaks and what remediation coverage is required. The curator then addresses the supply problem: locate, rank, deduplicate and assemble the smallest high-value multimodal training set that targets those measured gaps. Together they form a closed data-improvement loop: stress the policy, diagnose the weakness, curate the corrective data, retrain and re-evaluate.

| STRESS What breaks? | DIAGNOSE Why does it break? | CURATE What data fixes it? |
| :---: | :---: | :---: |

## **Industry / Technology References**

\[1\] NVIDIA Isaac Lab-Arena \- scalable, GPU-accelerated policy evaluation built on Isaac Lab. [Source](https://developer.nvidia.com/isaac/lab-arena)

\[2\] Databricks Unity Catalog \- governed storage and processing for unstructured files including images and video. [Source](https://docs.databricks.com/aws/en/unstructured/)

\[3\] Databricks MLflow \- experiment tracking for parameters, metrics, datasets, artifacts and model-development lineage; Models in Unity Catalog adds governed model lifecycle and lineage. [MLflow Tracking](https://docs.databricks.com/aws/en/mlflow/tracking) | [Model lifecycle](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/)

\[4\] AWS Guidance for Physical AI for Robotics \- Isaac Sim/Lab on GPU compute, scalable orchestration, S3 data loops and model iteration. [Source](https://docs.aws.amazon.com/solutions/physical-ai-for-robotics-on-aws/)

\[5\] LIBERO-plus \- official robustness benchmark implementation with systematic perturbations across seven dimensions; LeRobot provides current integration documentation. [Repo](https://github.com/sylvestf/LIBERO-plus) | [LeRobot docs](https://huggingface.co/docs/lerobot/main/en/libero_plus)

\[6\] OpenVLA-OFT \- canonical public repository, LIBERO checkpoints and native evaluation workflow; paper: Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. [Repo](https://github.com/moojink/openvla-oft) | [LIBERO guide](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) | [Paper](https://arxiv.org/abs/2502.19645) | [LeRobot status](https://github.com/huggingface/lerobot/issues/1082)

\[7\] NVIDIA Isaac GR00T N1.7 \- GA generalist VLA with official LIBERO checkpoints and current LeRobot integration. [Repo](https://github.com/NVIDIA/Isaac-GR00T) | [LIBERO checkpoint](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) | [LeRobot GR00T](https://huggingface.co/docs/lerobot/main/groot)

\[8\] NVIDIA Cosmos Embed1 \- joint video-text embeddings for semantic search, similarity and clustering of short-form Physical AI video. [Source](https://docs.nvidia.com/nim/cosmos-embed1/latest/introduction.html)

\[9\] LIBERO \- public lifelong robot-learning benchmark and manipulation task suites. [Source](https://libero-project.github.io/main.html)

\[10\] EXL \+ iMerit \- completed combination of EXL enterprise data/AI capabilities with iMerit model-training, evaluation and reinforcement-learning capabilities. [Source](https://ir.exlservice.com/news-releases/news-release-details/exl-completes-acquisition-imerit-accelerating-enterprise-ai)

\[11\] LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models \- primary paper for the perturbation taxonomy, 10,030 task instances, L1-L5 difficulty levels and sensitivity results. [Paper](https://arxiv.org/abs/2510.13626) | [Repo](https://github.com/sylvestf/LIBERO-plus)

&nbsp;