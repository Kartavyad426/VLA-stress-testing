# DATA AND BENCHMARKS

*Written 2026-09-17. The single reference for "what data and benchmarks exist, and which fit this
pipeline". **Data and benchmarks only — no models.***

> **Status: simulation half complete. The real-robot and failure-dataset half was stopped as
> redundant — §5–§6 now point at where that material already lives rather than duplicating it.** Marks: **[F]** read at full text or from installed source · **[A]**
> abstract/landing page · **[S]** search-snippet only · **[?]** unverified.
>
> A large fraction of this document is **[F] from the installed tree**
> (`third_party/lerobot/`, `.venvs/lerobot/`) rather than from the web, which makes it checkable and
> version-accurate for *our* install rather than for the latest release.

## What this pipeline needs

| # | requirement | why it is the binding constraint |
|---|---|---|
| a | runs on an **8 GB laptop GPU** | GR00T N1.7 is 3B params; F4 already killed π0.5 on this card |
| b | **privileged state** — object poses, eef-to-object, contact, success predicates | without it we get pass/fail and nothing minable. This is the one that eliminates most candidates |
| c | **settable perturbation knobs** | no knobs, no counterfactual attribution, no `ddmin` |
| d | **LeRobot support**, or a clear adapter path | |
| e | **reproducible published numbers** | needed to validate a setup before trusting a result |
| f | checkpoints for SmolVLA / MINERVA / GR00T N1.7 | |

> **Requirement (b) is the filter that matters, and it has a subtlety that cost this project real
> time.** The question is not whether the simulator *has* privileged state — MuJoCo and SAPIEN both
> do. It is whether that state is reachable **through the wrapper stack a policy actually runs
> under**. LeRobot's LIBERO wrapper extracts exactly 7 keys from robosuite's raw observation and
> discards the rest (`envs/libero.py:291-302`); we recover object state only by reaching past it to
> `sim.data`. **Every "yes" in the privileged-state column below means "reachable through the
> wrapper", not "exists in the simulator".**

---

## 0. Comparison table

| benchmark | sim | LeRobot env | privileged state | perturbation knobs | our policies | 8 GB | verdict |
|---|---|---|---|---|---|---|---|
| **LIBERO** | MuJoCo / robosuite | ✅ `libero` | ✅ via `sim.data` | ✗ (we add our own) | SmolVLA ✅ MINERVA ✅ **GR00T ✅** | ⚠ *(GR00T VRAM)* | **in use; validated** |
| **LIBERO-Plus** | MuJoCo / robosuite | ✅ `libero_plus` | ✅ **identical to LIBERO** | ✅ **7 dims, 10,030 instances** | loadable ✅ / no published numbers | ⚠ *(GR00T VRAM)* | **ADD NEXT — §7** |
| **LIBERO-PRO** | MuJoCo / robosuite | ✗ | ✅ (same base) | ✅ **generator ships** | none | ✅ | strong second; needs adapter |
| **LIBERO-Safety** | MuJoCo / robosuite | ✗ | ✅ (same base) | ✅ 3 difficulty tiers | none | ✅ | only if safety-cost work starts |
| **RoboCasa365** | MuJoCo / robosuite | ✅ `robocasa` | ✅ robosuite `_check_grasp` | ⚠ asset/scene randomisation | **SmolVLA ✅** | ⚠ | **best breadth option** |
| **MetaWorld** | MuJoCo | ✅ `metaworld` | ✅ | ✗ | none | ✅ | toy-scale; low value now |
| **VLABench** | MuJoCo | ✅ `vlabench` | **[?]** | ⚠ strong per-task randomisation | none | **[?]** | interesting for language work |
| **RoboTwin 2.0** | SAPIEN | ✅ `robotwin` | **[?]** | **[?]** | none | **[?]** | dual-arm; wrong embodiment |
| **RoboMME** | ManiSkill / SAPIEN | ✅ `robomme` | **[?]** | ✗ | none | **[?]** | memory benchmark; off-thesis |
| **IsaacLab-Arena** | Isaac Sim | ✅ `isaaclab_arena` | **[?]** | ✅ | none | ✗ **almost certainly** | Isaac Sim on 8 GB is implausible |
| **SimplerEnv** | SAPIEN / ManiSkill2 | ✗ | **[?]** | ✅ variant aggregation | **GR00T ✅** | **[?]** | **GR00T's own eval env — §7** |
| **ManiSkill 3** | SAPIEN | ✗ (indirect) | ✅ | ✅ | none | ✅ *(2–4× less GPU mem)* | good sim, no VLA story |
| **COLOSSEUM** | CoppeliaSim / PyRep | ✗ | ✅ | ✅ **14 axes** | none | **[?]** | best knob design in the field |
| **CALVIN** | PyBullet | ✗ | ✅ | ⚠ 4 env splits | none | ✅ | superseded for our purposes |

---

## 1. The LIBERO family — what we already use

### 1.1 LIBERO (in use, validated)

Four suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) of 10 tasks each, plus
`libero_90`. MuJoCo + robosuite 1.4.0, Franka Panda. **[F] from the installed tree.**

*Privileged state:* the reason this benchmark works for us. Object poses, contacts and joint state are
reachable via `sim.data`, and robosuite exposes a **standard bilateral-fingerpad grasp check** —
`ManipulationEnv._check_grasp(gripper, object_geoms)` at
`robosuite/environments/manipulation/manipulation_env.py:201` **[F]**, used by every robosuite
manipulation task. Note that **LIBERO's own BDDL registry has no grasp or holding predicate** — the
registry is `true, false, in, on, up, printjointstate, open, close, turnon, turnoff`, and
`InContactPredicateFn` is implemented but **commented out** (`libero/envs/predicates/__init__.py:8`)
**[F]**.

*Perturbation knobs:* **none native.** Ours are bolted on in `vla_harness`, and the camera path is
currently defective (see §8).

*Reference numbers:* MINERVA reproduced here at 95.3% vs 95.75% published on MuJoCo 3.3.2.

### 1.2 LIBERO datasets — four variants, and they are not interchangeable

| dataset | size / format | notes |
|---|---|---|
| `lerobot/libero` | **1.9 GB**, MP4, 256×256 | **1,693 episodes, 273,465 frames, 40 tasks, fps=10** **[F] from `meta/info.json`**. Features: 2 video streams, `observation.state` (8-dim), `action` (7-dim), `task_index`. **No object poses, no contacts.** |
| `HuggingFaceVLA/libero` | 69.9 GB, PNG | same content, unpacked frames |
| `physical-intelligence/libero` | — | openpi's conversion; **the SmolVLA paper names this as its training data** |
| `IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot` | — | GR00T's training data; no-op frames removed. Image orientation vs LeRobot's 180° flip **unverified [?]** |

**Three findings about `lerobot/libero` that bear on the retraining-default taxonomy** (see
`docs/RETRAINING_DEFAULT_TAXONOMY.md` §5b), all **[F]**, computed here:

1. **No privileged state in the training corpus at all.** Any "data gap" claim must therefore be
   expressed in the policy's observation space — eef pose, gripper, actions, images, instruction.
   This is a hard constraint on the manifest's evidence column.
2. **Grasp points are extremely tightly clustered**: across 1,617 demos with a detectable grasp
   moment, per-task standard deviation is **1.1–2.6 cm**, spanning ranges of about 5 cm. The demos
   contain almost no variation in where the object is grasped — a direct, countable explanation for
   LIBERO-Plus's finding that these models *"merely learned the positional information of the target
   objects"*.
3. **One instruction per task, 40 total.** There are no paraphrases, so an instruction-coverage family
   is **degenerate on LIBERO** for the same structural reason `language_grounding` is.

**Unresolved [?]:** the dataset declares `fps = 10`; our evaluation runs `--env.fps=20`. Whether
LIBERO's native control rate reconciles these, or the dataset is subsampled, is unestablished — and
`lerobot#4614` documents that a non-20 Hz `--env.fps` silently rescales delta actions.

### 1.3 LIBERO-Plus — **registered in LeRobot, and this is the headline finding**

7 perturbation dimensions (camera viewpoints, object layouts, robot initial states, language
instructions, lighting, background textures, sensor noise) over 21 sub-dimensions, **10,030 released
task instances** filtered down from 14,000 generated.

**It reuses our exact environment.** `LiberoPlusEnv(LiberoEnv)` at `envs/configs.py:729` **[F]**, and
the docstring is explicit:

> *"The gym interface is identical to LIBERO so this class reuses `LiberoEnv` entirely — only the
> registered name and default task suite differ."*

The only behavioural difference is a flag threaded into `get_task_init_states`
(`envs/libero.py:67-72`) **[F]**. So privileged state, the wrapper chain, `_check_grasp`, and our
existing adapter all carry over unchanged.

*Install* (`docker/Dockerfile.benchmark.libero_plus`) **[F]** — and it does **not** coexist with our
current install:
- pins `robosuite==1.4.1` (we have 1.4.0), `bddl==1.0.1`, `gym==0.26.2`
- **pins `mujoco==3.7.0`**, which is **inside F2's broken range (3.4.0–3.8.1)** — see §8
- clones `sylvestf/LIBERO-plus` at SHA `4976dc3`, installs `--no-deps -e .`, and
  **uninstalls `hf-libero`**, which we depend on
- adds the clone to `PYTHONPATH` as a namespace package, shadowing `libero`
- downloads `assets.zip` from `Sylvest/LIBERO-plus`

⇒ **separate venv, and the MuJoCo pin must be overridden.**

> ### ⚠ THREE DIFFERENT MuJoCo VALUES ARE IN PLAY FOR THIS ONE BENCHMARK
>
> | source | value | status |
> |---|---|---|
> | LeRobot's documented install (`docs/source/libero_plus.mdx:55`) | **unpinned** → latest | **broken two ways** |
> | LIBERO-Plus's own Dockerfile | `3.7.0` | inside F2's broken physics range |
> | our recommendation | `3.3.2` or `3.3.7` | healthy |
>
> The documented install is `pip install -e ".[libero]" "robosuite==1.4.1" bddl easydict mujoco wand
> scikit-image gym` — **`mujoco` unpinned** **[F]**. That lands (a) past the **3.4.0 physics break**,
> and (b) potentially on **≥3.10.0, which changes the `mj_fullM` signature**; our installed robosuite
> calls it at `controllers/base_controller.py:156` **[F]**, and GR00T's `setup_libero.sh` pins 3.3.1
> for exactly this reason. Whether robosuite **1.4.1** fixed that call is **[?]**.
>
> **So: pin `mujoco==3.3.2` (or 3.3.7) in the LIBERO-Plus venv. The documented install is broken in
> two independent ways.** Credit: `vla-7f`.
>
> ### And a qualification to the recommendation itself **[?]**
>
> LIBERO-Plus's perturbation assets **and its difficulty filtering** were presumably produced under its
> own `3.7.0` pin. Overriding to 3.3.x is right for *physics* — but the difficulty labels, and
> possibly the lighting and camera dimensions that sit on the **3.3.3 rendering change**, may not
> carry over exactly.
>
> **There may be no MuJoCo version that is simultaneously correct for LIBERO's physics and faithful to
> LIBERO-Plus's own calibration.** Bounded impact, but it should be stated:
> - **Unaffected:** the perturbation *instances* are still usable, and our miner still runs on them.
> - **Affected:** reason 4 of §7 — *"reproduce their published ordering as validation of our miner"*.
>   If they evaluated at 3.7.0 and we evaluate at 3.3.7, a mismatch cannot be cleanly attributed to
>   our miner rather than to the version difference. The ordering is probably robust enough to survive
>   it, but "probably" is doing work there and the comparison is no longer clean.
> - **Cheap mitigation:** run one suite at both versions and check whether the ordering moves. That is
>   a small controlled experiment and it converts a `[?]` into a measurement.

*Reference numbers:* Table 1 of [arXiv:2510.13626](https://arxiv.org/abs/2510.13626) gives per-dimension
success for ten checkpoints **[F]** (quoted in full in `FAILURE_MINING_METHODS.md` §3.3). **None of
our three policies is among them** — no SmolVLA, MINERVA or GR00T row.

*Trap:* the released 10,030 are **difficulty-filtered** — instances solved by all-or-most of four
reference models were deleted. A rate over that corpus is not an unbiased robustness rate **[F]**.

### 1.4 LIBERO-PRO — the generator that actually ships

Four dimensions (objects, initial states, instructions, environments). Its headline: models above 90%
on standard LIBERO **collapse to 0.0%** under its generalized setting **[S]**.

**Its repo ships `perturbation.py` and an `evaluation_config.yaml`** with boolean knobs
(`use_swap`, `use_object`, `use_language`, `use_task`) and **intensity levels x0.1–x0.5**
**[A — README-level; not cloned or run]**. That is a knob API *with a magnitude axis*, which
LIBERO-Plus's release does not provide, and it covers **object-swap and language** — the two probes
LIBERO-Plus's corpus lacks.

No LeRobot env registration; would need an adapter.

### 1.5 LIBERO-Safety

5 suites × 3 difficulty tiers (L0–L2), 500+ tasks, consequence-oriented: affordance-aware grasping,
human-robot interaction, tabletop spatial avoidance, free-space hand-object avoidance, semantic safety
reasoning **[A]**. Metrics include collision rate and log-dimensionless jerk.

*Relevance:* it is the only published benchmark that would populate our `COST_SAFETY` level, which is
currently defined and unreachable (`FAILURE_FAMILY_AUDIT.md` §1). Not a priority until safety-cost
work starts.

---

## 2. Other benchmarks LeRobot registers — **[F] from `envs/configs.py`**

Ten env types are registered: `aloha`, `pusht`, `libero`, `metaworld`, `robocasa`, `vlabench`,
`isaaclab_arena`, `libero_plus`, `robotwin`, `robomme`. A `Dockerfile.benchmark.robocerebra` also
exists with **no corresponding registration** — newer or in progress **[?]**.

### 2.1 RoboCasa365 — the best breadth option

365 tasks (~65 atomic, ~300 composite) across 2,500 kitchen environments, 3,200+ assets, 600+ hours of
demonstrations, on a **PandaOmron 12-DOF mobile manipulator** **[F]** from LeRobot's docs.

- **robosuite/MuJoCo-based**, so `_check_grasp` and `sim.data` access transfer **directly from our
  LIBERO work** — the single biggest fit advantage after LIBERO-Plus.
- **`lerobot/smolvla_robocasa` checkpoint exists** **[F]** — one of our three policies.
- Observations: 16-dim state, **three** 256×256 cameras. Actions: 12-dim (base 4 + mode 1 + eef pos 3
  + eef rot 3 + gripper 1). **Different embodiment from LIBERO** — mobile base, three cameras.
- Task groups exposed as `--env.task` shortcuts: `atomic_seen`, `composite_seen`, `composite_unseen`,
  `pretrain50/100/200/300`.
- Protocol: **20 episodes per task**, stated as matching published results.

*Install traps* **[F]**: not on PyPI; RoboCasa's `setup.py` **hardcodes `lerobot==0.3.3`**, conflicting
with the repo, so it installs `--no-deps` from a git clone. Optional objaverse/aigen asset packs are
**~30 GB**. Without them, the env must stay on the `lightwheel` registry or it crashes with
`Probabilities contain NaN`.

*Verdict:* the strongest option if breadth and a second embodiment are wanted. Costs a new adapter
(3 cameras, 12-dim action, mobile base) but **not** new privileged-state work.

### 2.2 MetaWorld · VLABench · RoboTwin · RoboMME · IsaacLab-Arena

| | scale / notes **[F]** |
|---|---|
| **MetaWorld** | `metaworld-push-v2` default, `episode_length=400`, `multitask_eval` flag. MuJoCo. Small single-arm tasks; no language. Low value given LIBERO already works. |
| **VLABench** | `episode_length=500`, configurable `robot` and `action_mode`. 100 task categories, 2,000+ objects, instructions with **implicit intent rather than templates**, world-knowledge tasks **[A]**. The most interesting option for language-grounding work, since LIBERO is degenerate there (§1.2). Privileged state **[?]**. |
| **RoboTwin 2.0** | SAPIEN, **50 dual-arm tasks**, Aloha-AgileX 14-DOF, `episode_length=1200`, 3 cameras. Wrong embodiment for our policies. |
| **RoboMME** | ManiSkill/SAPIEN, 16 tasks over Counting/Permanence/Reference/Imitation, dataset `lerobot/robomme` (1,600 episodes). A **memory** benchmark — off-thesis. |
| **IsaacLab-Arena** | Isaac Sim. Config exposes `num_envs`, `embodiment`, `mimic`, `teleop_device`, `enable_cameras`, `headless`. **8 GB feasibility: almost certainly not** — Isaac Sim's own requirements exceed it. Would need the free-tier 16 GB path. |

---

## 3. Simulation benchmarks **not** in LeRobot

### 3.1 SimplerEnv — GR00T's own evaluation environment

Real-to-sim evaluation for **Google Robot** and **WidowX+Bridge** setups, CoRL 2024, built on
**SAPIEN / ManiSkill2** **[A]**. Two protocols: **Visual Matching** (overlay real images onto sim
backgrounds, match textures) and **Variant Aggregation** (average across background, lighting,
distractor and table-texture variants — *this is a perturbation knob set in all but name*).

**Why it matters here:** `vla-7f` found GR00T's LeRobot preprocessor embodiment map contains
`simpler_env_google: 0`, `simpler_env_widowx: 1`, `libero_sim: 2`, `droid_sim: 3` **[F]**. So
**SimplerEnv is one of GR00T's first-class eval environments**, and GR00T is the project priority.

*Against it:* no LeRobot env registration; **different simulator (SAPIEN/PhysX, not MuJoCo)**, so none
of our robosuite privileged-state work or MuJoCo version findings transfer — `vla-7f`'s point, and it
is a real cost. Privileged-state reachability **[?]**. Repo is forked widely
(`simpler-env/SimplerEnv`, `DelinQu/SimplerEnv-OpenVLA`, `allenzren/SimplerEnv`) with no single
canonical VLA-capable version **[S]** — a provenance hazard of exactly the kind §8 catalogues.

### 3.2 ManiSkill 3

SAPIEN, GPU-parallel simulation **and** rendering, 12 task domains, **up to 30,000+ FPS** and
**2–4× less GPU memory than comparable platforms** **[A]**, ICLR 2025, still **beta**.

Best raw simulator on this list for our hardware, and privileged state is first-class. But there is no
VLA checkpoint story for our three policies and no LeRobot env — it would be a platform bet, not a
benchmark addition.

### 3.3 COLOSSEUM — the best perturbation design in the field

20 RLBench tasks on **PyRep/CoppeliaSim**, with **14 perturbation axes** **[S]**: manipulation-object
colour/texture/size, receiver-object colour/texture/size, light colour, table colour/texture,
distractors, background texture, camera pose, and **object friction and mass**.

*The physical-property axes (friction, mass) are the only ones on this list we could not synthesise
ourselves*, and they are the axes most likely to matter on real hardware.

*Published result worth carrying regardless of whether we adopt it:* across 5 SOTA models success
degrades **30–50%** per factor, and **≥75% when factors are applied in unison** **[S]** — independent
corroboration of LIBERO-Plus's negative compositionality finding.

*Against:* CoppeliaSim is a heavy, separate simulator stack; no LeRobot support; no checkpoints for our
policies.

### 3.4 CALVIN

PyBullet, 34 tasks, four environment splits (A/B/C/D) for zero-shot transfer, long-horizon
language-conditioned chains. Superseded for our purposes: LIBERO covers the same ground with a
simulator we have working and privileged state we already reach.

---

## 4. Robustness / perturbation benchmarks — consolidated

| benchmark | base | knobs | magnitude axis | ships a generator |
|---|---|---|---|---|
| **LIBERO-Plus** | LIBERO | 7 dims / 21 sub | ✗ *(fixed corpus)* | ✗ |
| **LIBERO-PRO** | LIBERO | 4 dims | ✅ x0.1–x0.5 | ✅ |
| **COLOSSEUM** | RLBench | **14 axes** incl. friction/mass | ✅ | ✅ |
| **SimplerEnv** *(variant aggregation)* | SimplerEnv | 4-ish | ✗ | ⚠ |
| **ours** (`vla_harness`) | LIBERO | camera, ee offset, light | ✅ | ✅ — **but defective, §8** |

---

## 5. Real-robot datasets — **not surveyed here; already covered elsewhere**

Work on this section was **stopped on 2026-09-17** after a redundancy check, and the check was right:
the material is already in the repo, and re-surveying it would have duplicated existing coverage.

| dataset | where it is already covered |
|---|---|
| **Open X-Embodiment / RT-X** | `docs/LANDSCAPE.md` — [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) at §801, §1172, §1389, with a per-robot transfer-breakdown claim flagged unverified at §1433 |
| **DROID** | `docs/LANDSCAPE.md:441` — a crowd-sourced double-blind pairwise policy-comparison benchmark built on DROID across seven policies |
| **BridgeData / WidowX** | `docs/LANDSCAPE.md:458` — public evaluation cells; also throughout `SIM_TO_REAL_TRANSFER.md` §1 as SIMPLER's second setup |

> **The one fact from this area that most affects us is already written down**, in
> `FAILURE_MINING_METHODS.md:304`: **OXE contains essentially no failures.** For a project whose input
> is failures, that single line settles most of what a real-robot dataset survey would have concluded
> — these corpora are training data, not failure corpora, and they cannot supply what our mining layer
> consumes.

**Open, and genuinely unanswered [?]:** whether the `droid_sim: 3` entry in GR00T's preprocessor
embodiment map corresponds to a runnable simulated DROID environment. `vla-7f` was chasing this when
work stopped. It matters only if GR00T's DROID checkpoint becomes relevant.

## 6. Failure and recovery datasets — **covered in the methods survey, not duplicated**

`FAILURE_MINING_METHODS.md` §3 covers these **mechanistically**, which is the more useful treatment:

- **RoboFAC** — 9,440 erroneous trajectories, 78,623 QA pairs, 3-level/6-leaf taxonomy; and the
  finding that it scores **0.51 macro balanced accuracy on FailBench** — chance — despite 79.10% on
  its own benchmark.
- **AHA / FailGen** — 7 procedurally-injected failure modes; labels are ground truth about the
  *injection*, not about the failure.
- **FailBench** — 2,197 pooled attempts from 14 sources, **75% naturally occurring**; best VLM judge
  0.77 macro balanced accuracy, <0.60 on contact-rich.
- **RoboFail** — expert-annotated, used as the ground truth for the unsupervised-taxonomy evaluation.

### 6.1 The 2026 recovery subfield — **answered: method-only**

All five papers (FLARE, RedFlow, RePO-VLA, B2FF, FAR) are **method-only — no code or data links**
**[F]**, `vla-7f`. That closes the open question from when the survey stopped. Two define benchmarks
that are **not released**: RePO-VLA's **FRBench** (RoboTwin, 23,453 episodes, 46 tasks) and B2FF's
**failure-injected LIBERO** (Appendix E).

> **B2FF's protocol is the actionable one**, and it is the most immediately usable item found: it
> injects one perturbation at a sampled failure time **on LIBERO**, reporting success **56.3% → 74.0%**.
> Our harness runs LIBERO, so **we could reproduce the protocol without the dataset.** That is a
> published failure-injection design on our own benchmark — worth more to us than most released
> corpora, because it needs no new stack.

### 6.2 Candidate external fixtures — ranked by value to us

The motive throughout is the one hole nothing else fills: **family assignment has no ground truth**,
and κ cannot supply it at any threshold (`FAILURE_MINING_METHODS.md` §3.5).

**1. ViFailback** — [arXiv:2512.02787](https://arxiv.org/abs/2512.02787), **CVPR 2026**. Real ALOHA
failure *diagnosis and localisation* data, **MIT licence**, **with a LeRobot v2.1 conversion** **[S]**.
**The most immediately usable item on this list and new to our docs**: peer-reviewed, licensed,
LeRobot-format, real robot. Label schema unchecked **[?]**. *If anything here becomes a fixture, this
is the first candidate to examine.*

**2. `samwang1010/robotwin-failure-recovery`** — the schema is close to ideal and **it is not a
release** **[F]**, `vla-7f`. No README, no licence, every event flagged
`backup_is_not_formal_release: true`. 4,655 events, 13 tasks, ~240 GB.

*What it has that nothing else does:* extensive sim state (finger contact counts before/after,
`contact_lost`, object-to-TCP translation, per-object displacement, `s_pre`/`s_fail` snapshots,
`state_trace.json`), programmatic injected labels behind a five-stage acceptance funnel
(`labels_trusted: true`), and — **the part that changes what it is useful for** —
`fault_detected_frame` and `fork_frame`, i.e. **onset-localisation ground truth as well as cause**.

> **This is a stage-2 fixture more than a stage-3 one, and that is the more valuable gap.**
> `FAILURE_MINING_METHODS.md` §2.10 identifies failure-onset ground truth as *"the deepest issue in
> the whole stage"* — it is privileged, and on a real robot often cannot be constructed at all. A
> corpus that ships it is rarer than one shipping category labels.

*Why it is nonetheless not our family fixture*, and both reasons are structural:
- The faults are **injected into scripted experts, not produced by a VLA** — the §3.2 limitation
  exactly, against FailBench's finding that **75% of real failures occur naturally**.
- The six modes (`regrasp`, `orientation_recovery`, `relocalize`, `single_support_recovery`,
  `external_disturbance_recovery`, `handover_retry`) are **recovery *routes*, not a failure
  taxonomy** — they name the fix, not the fault.

Also RoboTwin-native HDF5 rather than LeRobot, dual-arm, no checkpoints for our policies.
**Verdict: do not build on an unlicensed backup. Watch for a formal release.**

**3. RoboMIND `failure_data/`** — real robot, native HDF5, **no sim state**, labels as Chinese
directory names, ~1,650 episodes over 11 categories, provenance undocumented and most plausibly
collectors binning teleop rejects **[F]**, `vla-7f`. Several categories are **data-quality rejects
rather than task failures**. Not a fixture — but a **useful external taxonomy comparison**, and §6.3
is why.

### 6.3 What RoboMIND's labels actually are — **corrected**

> **CORRECTION.** A first version of this section reported `jerky motion` (557 of ~1,650, the largest
> category) as evidence that policies need a trajectory-quality family we lack. **That was
> overstated.** `vla-7f` pointed out that **RoboMIND is a teleoperation collection and `failure_data/`
> sits next to its demonstrations** — and that `jerky motion`, `too fast`, `non-standard motion` and
> `gripper leaves frame` read as **rejection reasons for human demonstrations**, not policy failures.
> Their first message had already flagged that several categories are data-quality rejects; I had that
> caveat and did not apply it to the headline.

RoboMIND's released `failure_data/` distribution, largest first: **jerky motion 557**, regrasp 284,
unnecessary contact 232, collision-before-grasp 217, target displaced 118, failed placement 113, plus
five smaller — **~1,650 total, against a paper-stated 5k** (see below). Who applied the labels is
undocumented **[?]**.

**The corrected reading, which is narrower and more useful than the one it replaces:**

> **Trajectory quality as a *policy* failure family is suggested by LIBERO-Safety**, which names
> *"sub-optimal trajectory synthesis"* and *"erratic or overly conservative manoeuvres"* as a
> top-level mode **[A]**. That is one source, and it is about policy behaviour.
>
> **RoboMIND shows something different: that data collectors treat trajectory quality as a
> first-class reject reason for demonstrations.** That is evidence about **what makes demonstration
> data usable — a data-gap concern — not about how policies fail.**

**And the second reading is the one that pays** — but **not with a number.**

> **SECOND CORRECTION, same section.** I wrote that collectors reject "on the order of a third of
> teleop for motion quality alone", from 557 of ~1,650. **That is a denominator error.** 557/1,650 is
> jerky motion's **share of the rejects**, not a rejection rate. A rejection rate needs total episodes
> collected — kept *plus* rejected. Caught by `vla-7f`; it is exactly the ratio-without-a-denominator
> mistake that has been my most reliable failure mode this week.

Verified against the paper **[F]**, [arXiv:2412.13877](https://arxiv.org/abs/2412.13877): RoboMIND is
**107k demonstration trajectories, 479 tasks, 96 object classes**, four embodiments. So against the
real denominator, the released `failure_data/` is **~1.5% of collection** and jerky-motion entries are
**~0.5%** — the opposite of a third.

**The verification also turned up a discrepancy neither of us had.** The abstract states the dataset
*"includes **5k** real-world failure demonstrations, each accompanied by detailed causes, enabling
failure reflection and correction during policy learning"* **[F]** — but `vla-7f` counted **~1,650**
in the released `failure_data/`. **The release appears to be roughly a third of the stated failure
corpus.** That converts *"the subset may not be exhaustive"* from a caution into a measured
discrepancy, and it means **no rejection rate can be derived from the release at all.**

It also qualifies the "teleop QA bin" reading in both directions, and the honest position is that both
are partly right:
- **For it:** the category names (`jerky motion`, `too fast`, `gripper leaves frame`) read as
  demonstration-quality criteria, and the failures are **human-generated**, not policy-generated.
- **Against it:** the paper frames them as a **deliberate resource** — *"failure demonstrations, each
  accompanied by detailed causes"* — not as a discard pile. Who applied the labels remains
  undocumented **[?]**.

**What survives, and it is still worth having:**

1. **Data collectors treat motion quality as a first-class reject criterion**, so a manifest spec
   should **name its quality criteria** rather than only a count and a condition. That is a change to
   the spec field and it needs no number.
2. **Jerky motion is the most common reason within that set** — 557 of ~1,650.

**What does not survive: any figure for how much rejection inflates collection cost.** RoboMIND
publishes no rejection rate, and the released subset cannot supply one. If that number is ever wanted,
it has to come from a client's own collection pipeline — which is the same shape as the prevalence
argument in `SIM_TO_REAL_TRANSFER.md` §5: a quantity only the deployment can supply.

**The same caution retracts a tension I drew.** I wrote that the three categories we lack —
`collision-before-grasp`, `unnecessary contact`, `target displaced` — are all contact/physics-mediated
and therefore exactly the class that does not transfer from simulation
(`SIM_TO_REAL_TRANSFER.md` §1.3). **The sim-to-real point stands on its own evidence. RoboMIND does
not support it**, because those categories are most likely teleoperator errors rather than policy
failures. Withdrawn as corroboration.

### 6.4 One thing that does *not* support family C

`robotwin-failure-recovery`'s `nominal_continuation` branch "never self-completes". **This is
tautological, not a finding** — `vla-7f` confirmed from the event manifest **[F]** that the branch is
the **scripted reference replayed** (`nominal_resume_reference_cursor: 429` immediately after
`fault_confirmed_reference_cursor: 428`, with checks named
`nominal_continuation_resumed_affected_arm_commands`). It replays the verified reference's commands
from the next cursor, so of course a replayed expert trajectory does not succeed from a perturbed
state.

**It says nothing about whether recovery is absent from VLAs**, and must not be cited for family C.
The claim that VLAs cannot recover rests on the published position that they are trained on
failure-free demonstrations — not on this.

Provenance: §6.1–§6.3 are `vla-7f`'s reading, relayed. Full notes in their scratch file. Items marked
**[F]** were read from the artefacts (including one `event_manifest.json` fetched by HTTP range
request); **[S]** items are listing-level.

---

## 7. Recommendation

### Add **LIBERO-Plus** next

The argument is that it costs almost nothing and fixes something already broken.

1. **Zero adapter work.** `LiberoPlusEnv` reuses `LiberoEnv` entirely **[F]**. Our env adapter, our
   privileged-state extraction, `_check_grasp`, the phase segmenter and the mining layer all carry
   over unchanged. Nothing else on this list is free.
2. **It replaces perturbation code that is currently invalid.** Our camera arms were withdrawn
   yesterday (§8). LIBERO-Plus supplies 7 published, peer-reviewed dimensions over 10,030 instances
   instead of three home-grown knobs, one of which was silently wrong for a month.
3. **It is the boundary-location instrument we lack.** Our own sweep has **no samples between 0° and
   ~13°** of actual camera change, which is where this policy's transition happens — so a fixed grid
   at 5/10/15/20 cannot locate the boundary. LIBERO-Plus's graded sub-dimensions and difficulty
   levels sample that region.
4. **Published per-dimension numbers for ten checkpoints** give the external ordering we have never
   had — camera and robot-state dominate, language is consistently smallest. If our miner reproduces
   that ordering on a new policy, that is real evidence it works.

**Costs, stated honestly:** a separate venv (it uninstalls `hf-libero` and shadows `libero`); the
`mujoco==3.7.0` pin must be overridden to a healthy version; the released corpus is difficulty-filtered
so absolute rates are not robustness rates; and **none of our three policies appears in its published
table**, so we would be extending it rather than reproducing it.

### GR00T — the loading question is settled, and it strengthens the case

> **CORRECTION, 2026-09-17.** An earlier version of this section said GR00T's LIBERO checkpoints were
> **not** LeRobot-loadable, citing commit `b198613`. **That is wrong.** `b198613` reasoned from
> NVIDIA's native Isaac-GR00T format and missed both the LeRobot policy type and NVIDIA's conversions;
> it was corrected on 2026-09-16 (`MODELS_AND_COMPUTE.md` §R1). I propagated the claim from the commit
> without checking it. Verified here **[F]**:
>
> - `third_party/lerobot/src/lerobot/policies/groot/` is a **first-class policy type** —
>   `GrootConfig` registered as `"groot"` at `configuration_groot.py:241-243`, imported in
>   `policies/factory.py`.
> - NVIDIA publishes **LeRobot-format** checkpoints `nvidia/gr00t17-lerobot-libero_{spatial,object,goal,10}-640`,
>   built against LeRobot 0.6.1.
> - LeRobot's groot docs report **95 / 100 / 98 / 93 (96.5% avg)** at n_episodes ≥ 50.

So **LIBERO and LIBERO-Plus are the GR00T-native, minable path.** There is no longer a trade-off
between "cheap and minable" and "GR00T-native" — they are the same option, which makes the §7
recommendation stronger rather than weaker. SimplerEnv drops from "the only GR00T-native choice" to
"a second GR00T eval environment we could add later".

**The two blockers that remain are different, and neither is about loading:**

1. **VRAM on 8 GB — unverified, and the precedent is discouraging.** GR00T N1.7 is ~3B parameters.
   **F4 established that π0.5, also ~3B, exceeds 8 GB on weights alone** — not contention, the model.
   `docs/LANDSCAPE.md:944` additionally records that **community reports say GR00T needs 16 GB**,
   which points the same way.
   Until someone loads GR00T on this card, treat 8 GB feasibility as **[?] leaning negative**, and
   note that this is a *hardware* question with a known answer path (the free-tier 16 GB route), not a
   research one.
2. **Image orientation.** GR00T trained on `IPEC-COMMUNITY/libero_*_no_noops` against LeRobot's 180°
   flip (`LiberoProcessorStep`). Unverified **[?]**, and it is exactly the silent-mismatch class
   catalogued in `docs/POLICY_SIM_COUPLING.md` — wrong orientation would degrade results without
   erroring.

**One config detail worth carrying**, since it is the same contract class: `GrootConfig` defaults to
`chunk_size = 40` and `n_action_steps = 40`, and LeRobot **auto-migrates GR00T N1.5-era values of 50 to
40** (`configuration_groot.py:398-406`) **[F]**. A silent value substitution of exactly the kind that
made `n_action_steps` hard to pin down for SmolVLA.

### If breadth is wanted later: **RoboCasa365**

robosuite/MuJoCo (privileged state transfers), a real SmolVLA checkpoint, 365 tasks, and a second
embodiment. Costs a genuine adapter — three cameras, 12-dim action, mobile base — and up to 30 GB of
assets.

### Do not pursue now

IsaacLab-Arena (8 GB), RoboTwin (wrong embodiment), RoboMME (off-thesis), MetaWorld (superseded by
LIBERO), CALVIN (superseded), ManiSkill 3 (platform bet with no VLA story), COLOSSEUM (heavy separate
simulator — but **steal its 14-axis design**, especially friction and mass).

---

## 8. Version traps and known reproduction problems

| trap | affects | detail |
|---|---|---|
| **MuJoCo ≥3.4.0 corrupts LIBERO init states** | LIBERO, LIBERO-Plus, LIBERO-PRO, LIBERO-Safety | 3.4.0's box-box collision fix leaves `libero_spatial` task 5's bowl tilted. Reporter's causal test: re-settling the init file alone takes SmolVLA 28% → 84% at fixed 3.8.1. Broken 3.4.0–3.8.1; healthy 3.2.7, 3.3.0, 3.3.7 |
| **LeRobot's own LIBERO-Plus Dockerfile pins `mujoco==3.7.0`** | LIBERO-Plus | **Inside the broken range** **[F]**. Must be overridden |
| **LeRobot's documented LIBERO-Plus install leaves `mujoco` UNPINNED** | LIBERO-Plus | `libero_plus.mdx:55` **[F]** — resolves to latest, landing past the 3.4.0 physics break and potentially on ≥3.10.0's `mj_fullM` change. Broken two independent ways |
| **LIBERO-Plus difficulty labels may be version-bound** | LIBERO-Plus | Assets and filtering produced under 3.7.0; overriding to 3.3.x may not preserve them, especially lighting/camera on the 3.3.3 rendering change **[?]** |
| **MuJoCo ≥3.3.3 rendering shift** | all MuJoCo benchmarks | Object-floor rendering changed; MINERVA pins 3.3.2 for this reason |
| **`pip install lerobot[libero]` resolves to 3.8.1** | LIBERO | LeRobot pins `mujoco<3.9.0`, which guards API breaks and not behavioural ones |
| **MuJoCo 3.10.0 crashes robosuite 1.4.0** | all robosuite benchmarks | `mj_fullM` signature change **[S]** |
| **None of the above applies to SAPIEN** | SimplerEnv, ManiSkill, RoboTwin, RoboMME | Different physics stack; **needs its own version audit** |
| **Init states chosen by a reset counter, not the seed** | LIBERO | |
| **`goal` task 5's target is a MuJoCo site** | LIBERO | |
| **`libero_goal` traces here are 77% missing `_gt_eef_to_object`** | our campaign | Absent data, not a re-mine; needs re-capture |
| **Our camera perturbation is defective** | our campaign | `if yaw or pitch or dist:` makes `yaw=0.0` skip the block, so the control never runs the re-aim every treated arm carries. A "5° yaw" is a ~13° rotation plus 6–8 cm translation, measured from recorded extrinsics **[F]**. All 12 `camera_misalignment` rows invalid |
| **`reset()` returns a stale frame** | our campaign | First observation of every perturbed episode shows the unperturbed scene |
| **RoboCasa hardcodes `lerobot==0.3.3`** | RoboCasa | Install `--no-deps` |
| **RoboCasa crashes without asset packs** | RoboCasa | `Probabilities contain NaN` unless restricted to the `lightwheel` registry |

---

## 9. What could not be verified

| claim | mark | what would settle it |
|---|---|---|
| Privileged-state reachability for VLABench, RoboTwin, RoboMME, IsaacLab-Arena, SimplerEnv | **[?]** | read each env wrapper's observation extraction, as was done for LIBERO |
| 8 GB feasibility for VLABench, RoboTwin, RoboMME, SimplerEnv | **[?]** | install and run one episode |
| LIBERO-PRO generator actually runs | **[A]** | clone and execute `perturbation.py` |
| `IPEC-COMMUNITY` image orientation vs LeRobot's 180° flip | **[?]** | compare frames directly |
| `lerobot/libero` fps=10 vs our `--env.fps=20` | **[?]** | check LIBERO's native control rate |
| `robocerebra` benchmark | **[?]** | Dockerfile exists, no env registration |
| COLOSSEUM's 30–50% / ≥75% degradation figures | **[S]** | snippet-level only |
| Whether any of our three policies has a LIBERO-Plus number | **[F] — no** | none appears in Table 1 |
| **GR00T N1.7 fits in 8 GB** | **[?] leaning negative** | load it. ~3B params, and F4 killed π0.5 at the same scale on weights alone |
| GR00T image orientation vs LeRobot's 180° flip | **[?]** | compare a training frame against a rendered eval frame |
