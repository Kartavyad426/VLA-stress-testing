# POLICY / SIMULATOR COUPLING — do VLA policies ship with a simulator, and what silently breaks LIBERO?

*Written 2026-09-16 by session `vla-7f`, in collaboration with `vla-81` (taxonomy audit) and `my primary`
(conformance gate, campaign). Answers five questions put by `my primary` while restructuring each policy into a
swappable box. Nothing here was run on the GPU; every local measurement is CPU-only over cached artifacts.*

> **Marks.** **[F]** full text / raw file retrieved, numbers read off the source · **[A]** abstract, landing page or
> tool summary only · **[S]** search snippet only · **[?]** could not verify; listed because the lead is worth chasing.
> "Local" = read in `third_party/lerobot` (HEAD `b6ec006`, the editable install behind `.venvs/lerobot`), in
> `.venvs/lerobot/.../site-packages`, or in the HF cache. Community issue numbers are single reporters unless
> stated; none of the GitHub evidence is peer reviewed.

---

## 0. Headline findings

1. **Policies do not have MuJoCo requirements; training data and benchmark assets do.** A single pinned shared
   simulator is viable. No primary source shows two published LIBERO checkpoints needing conflicting versions (§3).
   The pin must sit **below 3.4.0** (physics) and — for policies trained on the standard renders — **at or below
   3.3.2** (lighting). Every exact pin in the field is in 3.2.3–3.3.2.
2. **There is no standard conformance check, and LeRobot's pre-eval check is name-only by construction.**
   `policies/factory.py:426-428` + `policies/utils.py:249-257` compare visual key *names* only — never shapes; state
   and action checks are a literal `TODO`; passing any `--rename_map` skips the check entirely. This is the
   mechanism by which F9's 360-vs-256 render mismatch passed silently (§4).
3. **`config.json` is not the eval contract.** In 2 of 4 target checkpoints the documented eval overrides the
   shipped `n_action_steps`; in the other 2 no eval is documented at all. Flip, camera mapping, control mode,
   render resolution and simulator version are declared by **none** of them (§5). A correct check must inspect
   the values **live on the policy/env objects at rollout time** against a hand-curated per-checkpoint table,
   because no machine-readable source exists.
4. **`HuggingFaceVLA/smolvla_libero` is not the SmolVLA paper's headline model.** It is a 32-layer VLM + 32-layer
   expert, width 0.5, 604.9M-parameter build — the paper's *ablation baseline* architecture, not the 16-layer /
   0.75-width 0.45B model behind Table 2's 87.3%. No published LIBERO number exists for this checkpoint (§6).
   This reframes F8. Escalated by `my primary` as `PENDING_DECISIONS.md` #20.
5. **Against community reproductions of the same checkpoint, our residual is concentrated in `libero_goal`**
   (ours 68 vs 81/83/87), and every community run used MuJoCo 3.3.2 where we use 3.3.7 (§7).

---

## 1. Q1 — how LIBERO-evaluating projects are distributed

**Two conventions, neither ships a reproducible simulator by default.** Exact MuJoCo pins are the exception.

| Project | Distribution | LIBERO source | robosuite | MuJoCo | Docker |
|---|---|---|---|---|---|
| OpenVLA (`c8f03f4`) | ckpt + partial reqs | upstream master, **unpinned** (README:519-523) [F] | ==1.4.1 (`libero_requirements.txt:2`) [F] | **not documented** | no |
| OpenVLA-OFT (`e4287e9`) | ckpt + partial reqs | upstream master, unpinned (LIBERO.md:24) [F] | ==1.4.1 [F] | **not documented** | no |
| MiniVLA (`0822b36`) | ckpt only | upstream master, unpinned [F] | **unpinned** [F] | **not documented** | no |
| VLA-Adapter (`23fa0c9`) | ckpt + partial reqs | upstream master, unpinned [F] | ==1.4.1 [F] | **not documented** | no |
| openpi (main) | ckpt + `uv pip compile` lockfile + compose | submodule @ `f78abd68` (2023-12) [F] | ==1.4.1 (`examples/libero/requirements.txt:90`) [F] | **==3.2.3** (`:43`) [F] | **yes**, digest-pinned base, built locally |
| SmolVLA / LeRobot (`b6ec006`) | ckpt + pip extra | **fork** `hf-libero>=0.1.4,<0.2` (pyproject:279) [F] | ==1.4.0 (hf-libero METADATA:16) [F] | range `>=3.0,<3.9` (METADATA:28); **uv.lock resolves 3.8.1** (`uv.lock:4302`) [F] | CI image only, floating tag |
| GR00T N1.7 (`51d4c89`) | ckpt + setup-script venv | submodule @ `8f1084e3` [F] | ==1.4.0 [F] | **==3.3.1** (`setup_libero.sh:56`) [F] | general image, LIBERO not included |
| MINERVA (`64c3cc8`) | ckpt @ HF revision + `uv sync --locked` | hf-libero 0.1.4 [F] | 1.4.0 (uv.lock) [F] | **==3.3.2** (pyproject:369 constraint + runtime assert) [F] | not the documented path |
| Octo (`241fb35`) | — | **no LIBERO eval**; its LIBERO numbers come from the OpenVLA paper [F] | — | — | — |

- Upstream LIBERO `setup.py:17` has `install_requires=[]`; `requirements.txt:10` pins `robosuite==1.4.0` and never
  mentions MuJoCo [F]. robosuite 1.4.0/1.4.1 require only `mujoco>=2.3.0` [F].
- **The OpenVLA family (4 of 8) is "bring your own LIBERO"**; the LeRobot route (SmolVLA, GR00T-LeRobot, MINERVA)
  is the growing second convention and bounds MuJoCo only by a range that includes known-broken versions.
- **Only openpi's documented default path is Docker; nobody publishes a prebuilt LIBERO eval image.**
- The three exact pins (3.2.3, 3.3.1, 3.3.2) all differ. Numbers from different projects were produced in
  different simulators.
- Our environment: mujoco **3.3.7**, robosuite **1.4.0**, hf-libero 0.1.4, lerobot 0.6.2 (local) [F].

---

## 2. Q2 — policy/simulator mismatches that silently degrade LIBERO

"Documented" = a primary source reports a specific success-rate change for that cause. "Plausible" = mechanism
exists, no number found. Rows marked **(config-loss)** are distinct from mismatches: the user specified a correct
value and the stack dropped it, so a declared-vs-passed comparison cannot catch them.

### 2.1 Documented with numbers

| # | Mismatch | Evidence | Status for our runs |
|---|---|---|---|
| 1 | **MuJoCo ≥3.4.0 physics** — box-box distance fix (commit `883836848a67`, `con[i].dist = 2*points[i][2]`) leaves `libero_spatial` task 5's stored init state settling with the bowl tilted on the ramekin rim | lerobot#4390 [F]: task 5 SmolVLA 80→28, OFT-GRPO 98→12, OFT-SFT 96→52, π0.5 90→86. **Causal test at fixed 3.8.1: re-settled init file alone restores SmolVLA to 84%.** Corroborated by LIBERO#141 [F]. Pin PR lerobot#4465 **open** | 3.3.7 — not affected |
| 2 | **MuJoCo ≥3.3.3 rendering/lighting** — darker LIBERO-Object floors | LIBERO#88 [F] (screenshots). openvla#282 [F]: OpenVLA Object **68 → 85.2** with only `mujoco==3.3.2`; second user ~70 → 84.8. oft#150 [F]: spatial 94 → 99. Counter-example oft#128 [F]: 56.6 → 58.0. MINERVA README:198-208 [F]: scratch CNN Object 96% → 0% at ≥3.3.5; dataset-vs-env pixel gap 1.9 → 34.5. Candidate cause: changelog 3.3.3 `3e9bc79b` sRGB textures — **our inference, unconfirmed** | **3.3.7 — inside the affected range.** Unmeasured for SmolVLA |
| 3 | **MuJoCo ≥3.10.0 crash** — `mj_fullM` signature change; robosuite 1.4.0 calls the old one (`controllers/base_controller.py:156`) | GR00T `setup_libero.sh:51-56` [F] | 3.3.7 — not affected |
| 4 | **`n_action_steps` = chunk size (50)** | SmolVLA paper Table 13 [F]: avg 51.8 at 50 vs 82.8 at 10 vs 80.3 at 1 (ablation models). lerobot#4614 [F]: paired McNemar, `lerobot/smolvla_libero` object −20 pp (p=0.031), spatial −23 pp (p=2e-7). `lerobot/smolvla_libero` ships 50; fix PR #4615 open | we use 10; published used 1 (paper §4.3 [F]); Table 13 spatial is 89 at both |
| 5 | **Render resolution ≠ training resolution** (LeRobot `LiberoEnv` config default 360; checkpoint declares and was trained on 256; policy then pads to 512) | F9 (ours, paired, n=100/suite): spatial **+17.0 pp** (p=0.011), object +5 (n.s.), goal −1, long +2. openpi `main.py:18` `LIBERO_ENV_RESOLUTION = 256 # resolution used to render training data` [F] | fixed in `res256` runs. Note the change also alters the 512 upsampling ratio and padding geometry, not just a nominal shape match |
| 6 | **Control frequency / `--env.fps` ≠ 20** — rescales how far each delta action moves | lerobot#4614 [F]: fps 10 → 0 successes on object; spatial fps 16/18/22/25 → 3/14/23/8 of 32. Easy to pick up by mistake: `lerobot/libero` metadata says fps 10, `lerobot/smolvla_libero` train_config says 30. Warning PR #4615 open | fps 20 (`baseline.log:61`, `res256.log:41`) [F] |
| 7 | **`--env.max_parallel_tasks > 1`** — one policy object shared across threads | lerobot#4327 [F]: 85% at 1 vs **0%** at 4. Still present locally (`scripts/lerobot_eval.py:1009-1012`, `:1054-1058`) [F] | 1 (`res256.log:45`) [F] |
| 8 | **Model-code / transformers drift for older checkpoints** | oft#135 [F]: stock transformers vs the OFT fork, 60.2 vs 98.1. lerobot#3638 [F]: `pi05_libero_finetuned` → 0% on libero_10 after a padding change (maintainer-confirmed). No SmolVLA report [?] | our checkpoint pushed 2025-09-17, evaluated on lerobot 0.6.2 / transformers 5.5.4. Plausible, unmeasured |
| 9 | **Which init states are visited** | #2850 / #4152 [A] (fixed locally, `envs/libero.py:386-391`, `envs/utils.py:186-224`). PR #4315 [F]: seed-derived states move spatial tasks 0–3 72.5 → 75.0 | batch_size 1 visits states 0..9, same as openvla/openpi [F] |

### 2.2 Silent config-loss (value specified correctly, then dropped)

| # | Mechanism | Evidence |
|---|---|---|
| 10 | **`--policy.pretrained_path` loads weights only** — `n_action_steps` and `empty_cameras` fall back to class defaults (50, 0) unless passed again | `docs/source/pi05.mdx:104` [F, local] |
| 11 | **Empty CLI `rename_map` default overwrites the checkpoint's saved one** (lerobot 0.6.1) | lerobot#4614 body, fix #4578 open [F]. Not ours: our preprocessor `rename_map` is `{}` |
| 12 | **`num_vlm_layers: 0` is a sentinel for "no truncation", not "zero layers"** — silently selects a different architecture (32 layers) from the class default (16) | `configuration_smolvla.py:96`, `smolvlm_with_expert.py:102-105, 112` [F, local]. See §6 |

### 2.3 Checked and matching, or ruled out

| # | Item | Finding |
|---|---|---|
| 13 | **`num_steps_wait`** (no-op settle steps after reset) | LeRobot MolmoAct2 docs assert 10 "does not reliably let the LIBERO scene stabilize and can degrade measured success" and use 50 (`docs/source/molmoact2.mdx:319-321`) [F, sentence only; no numbers]. **Not plumbed into `lerobot-eval`**: not a field of `LiberoEnv` in `envs/configs.py`; only `envs/libero.py:127,162,351` [F]. **Every reference harness, including those producing published numbers, uses 10** (openvla `run_libero_eval.py:72`, OFT `:113`, openpi `main.py:37`, VLA-Adapter `:109`, MINERVA fork `libero.py:127`); LIBERO's own harness uses **5** (`libero/lifelong/metric.py:119-122`) [F]; π0.5 scores 97.5% through the same LeRobot env at 10. The SmolVLA paper states no settle value [F, searched]. **Cannot explain a deficit relative to published results** |
| 14 | Image orientation (180° flip) | LeRobot `env_processor.py:59` `torch.flip(dims=[2,3])`; openvla `libero_utils.py:56`, openpi `main.py:115-116` use `[::-1, ::-1]` [F]. Matches. Horizontal-only flip PR lerobot#3882 closed unmerged |
| 15 | Gripper sign / binarisation | Checkpoint action stats gripper mean −0.0496, std 0.9988 → native LIBERO −1 open / +1 close [F, local]. openvla's `invert_gripper_action` exists only for its own RLDS loader. robosuite `PandaGripper.format_action` uses `np.sign` (`panda_gripper.py:55-56`) — only sign matters. Predicted values near 0 could chatter: plausible, undocumented |
| 16 | State representation | eef_pos(3) + axis-angle of xyzw quat(3) + gripper qpos(2) (`env_processor.py:68-75`), same as openvla `regenerate_libero_dataset.py:164-170` and openpi `main.py:136` [F] |
| 17 | **State frame / units / normalisation** | Checkpoint normaliser stats are **byte-identical to `lerobot/libero` `meta/stats.json`** [F, local]. Per-task first-frame `eef_pos`, all 377 dataset parquet files vs our campaign `rollouts.jsonl` t=0, 32 tasks, 33–50 dataset episodes each: **every task matches within ~1 cm**, with a uniform ~5 mm x offset in every suite [F, computed]. Global mean z = 0.76 is a mixture of scene heights (object scenes z ≈ 0.26), not a frame bug. **Ruled out.** A gate z-scoring live state against normaliser stats must do so per scene, or it false-alarms on object scenes (≈ −1.3σ in z) |
| 18 | Language instruction strings | All 40 local `task.language` strings match `physical-intelligence/libero` and `HuggingFaceVLA/libero` exactly [F, computed]. Ruled out |
| 19 | Action clipping | LeRobot rollout doesn't clip (`lerobot_eval.py:303-307`); robosuite does (`base_controller.py:120`) [F]. No effect |
| 20 | Episode length | LeRobot spatial 280 vs openvla/openpi 220; others match (`libero.py:98-104`) [F]. LeRobot is *more* lenient; cannot explain a drop |
| 21 | Center crop | Needed for openvla/OFT (trained with 90% random crop); SmolVLA has no crop augmentation [F]. N/A |
| 22 | JPEG compression | openvla eval JPEG round-trips to match RLDS (`libero_utils.py:42`); `physical-intelligence/libero` derives from that RLDS; LeRobot eval does not [F]. openpi also skips it and reaches 96.85 with π0.5 → likely small. Plausible |
| 23 | Precision / attention kernel | bf16 weights, float32 attention (`smolvlm_with_expert.py:94, 539-552`) [F]. openvla#282: flash-attn off 68 → 69 [F]. No SmolVLA data |
| 24 | robosuite 1.4.0 vs 1.4.1 | reference harnesses pin 1.4.1, hf-libero pins 1.4.0. No report found [?] |
| 25 | Seed | LeRobot reseeds each reset (`libero.py:342`); openvla fixed `env.seed(0)` with comment "seed seems to affect object positions even when using fixed initial state" (`libero_utils.py:24`); openpi 7 [F]. LeRobot maintainer: use seed 1000 (#2114) [F]. No isolated number |

### 2.4 Two rendering mismatches, on different axes

Do not merge these into one "rendering" category — they have **complementary suite profiles**:

| Mismatch | Axis | Suite profile |
|---|---|---|
| Lighting (MuJoCo ≥3.3.3) | photometric | MINERVA's dataset-vs-env pixel gap: object **34.3**, spatial 2.5, goal 1.6, long 0.7 [F, single author] |
| Resolution (360 vs 256) | geometric / sampling | F9: spatial **+17.0**, object +5 (n.s.), goal −1, long +2 |

A shape comparison catches the second and not the first. A **pixel-statistics comparison between rendered first
frames and the training dataset's stored first frames** catches both (§4.3).

---

## 3. Q3 — does MuJoCo / renderer version materially change LIBERO success?

**Yes — via two distinct, documented mechanisms, both properties of benchmark assets and training-data rendering,
not of policies.**

**(1) Physics-state, at exactly 3.4.0.** Details in §2.1 row 1. The damage is to one stored benchmark asset (task
5's init states) and is identical for every policy; policies differ only in how much a bad start hurts them (π0.5
−4 pp, SmolVLA −52, OFT-GRPO −86). No policy is *better* on ≥3.4.0, and re-settling the asset removes the effect
at any version. Reporter's settle probe: 3.2.7/3.3.0/3.3.7 bowl seated, 1.45 cm drift; 3.4.0/3.6.0/3.8.1 tilted,
3.72 cm, bit-identical over 50 states [F, numbers read off the issue, not re-run; one reporter].

**(2) Rendering/lighting, between 3.3.2 and 3.3.3/3.3.5.** Details in §2.1 row 2. What a checkpoint needs follows
**the version its training data was rendered with**. MINERVA additionally reports a kitchen-scene shift between
3.2.x and 3.3.x (EXPERIMENTS.md:447-450) and that LIBERO-90 needs 3.2.7 (`train_libero90.sh:36` refuses otherwise)
[F, single author, not peer reviewed]. MINERVA's scratch CNN is close to an *upper bound* on photometric
sensitivity — but openvla#282 shows a pretrained 7B VLA still losing ~15–17 pp on Object, so "pretrained backbones
absorb it" is **not** safe to assume.

**Also:** a crash hazard at 3.10.0 (§2.1 row 3), and hf-libero's `<3.9.0` cap (huggingface/LIBERO PR#3) and
lerobot PR#3751 guard API breaks only, not behaviour [F/A].

**Verdict on the architecture question: a shared pinned simulator is viable.** Every documented good setting fits
in 3.2.3–3.3.2 (openpi 3.2.3; #4390 healthy 3.2.7–3.3.7; GR00T 3.3.1; MINERVA 3.3.2, LIBERO-90 3.2.7). What each
box should carry is not its own simulator but a **declared training-data render version** plus per-task exposure
checks.

**Stated unknowns:**
- Which MuJoCo version rendered `lerobot/libero`, `HuggingFaceVLA/libero` or `physical-intelligence/libero` — **nobody documents this** [?]. MINERVA asserts 3.3.2 for its dataset; unverified.
- **No source compares EGL / OSMesa / GLFW backends.** oft#135 used OSMesa with low scores; nobody linked the two.
- **No peer-reviewed paper reports version sensitivity on LIBERO.** LIBERO-Plus (arXiv:2510.13626) and LIBERO-PRO (arXiv:2510.03827) appendices don't mention simulator versions [A]; MINERVA's LIBERO-Plus setup forces `mujoco==3.7.0` (`setup_libero_plus.sh:27`) [F], past both breaks.
- robosuite has no issue on version-dependent grasping or contact [S].
- The experiment that would most change this verdict — one checkpoint, 3.2.7 vs 3.3.2, per-suite — has not been published.
- MINERVA's arXiv ID (2609.03715) could not be confirmed from the card; one agent found an abstract page, one did not [?]. 2026, single author, not peer reviewed.

---

## 4. Q4 — is there a standard conformance check?

**No.** What exists, and what each actually checks:

### 4.1 LeRobot's pre-eval feature check — inadequate in a specific, documented way

`policies/factory.py:426-428` [F, local]:
```python
if not rename_map:
    validate_visual_features_consistency(cfg, features)
    # TODO: (jadechoghari) - add a check_state(cfg, features) and check_action(cfg, features)
```
`policies/utils.py:249-257` [F, local] builds sets of visual feature **names** and passes if *either* is a subset
of the other. Consequences:
- **Shapes are never compared** → 360×360 render against a 256×256 declaration passes (F9).
- **State and action are not checked** — a literal TODO.
- **Any `--rename_map` disables the check.** GR00T's LeRobot checkpoints declare `image`/`wrist_image` while the
  env emits `image`/`image2`; they work only via a hard-coded alias in `policies/groot/processor_groot.py:1579`,
  and with no match it logs once and feeds all cameras in alphabetical order.

### 4.2 Other pieces

| Piece | What it checks | Gap |
|---|---|---|
| LeRobot CI LIBERO smoke (`.github/workflows/benchmark_tests.yml:110-138`, PR #3319) [F] | one episode of `lerobot/smolvla_libero` runs; `parse_eval_metrics.py` records `pc_success` | **no threshold** — catches crashes, not accuracy |
| LeRobot unit tests (`tests/processor/test_libero_processor.py:57-72`, `tests/envs/test_dispatch.py`) [F] | key names, dtypes, shapes on random data; fps → control_freq; 8-dim state | no flip correctness against real data |
| Policy parity tests (π0.5, GR00T, XVLA vs original code) [F] | LeRobot port vs original numerics | **none for SmolVLA**; none involve the simulator |
| Published suite-level results (LeRobot π0.5 table, openpi README, SmolVLA Table 2) [A/F] | expected aggregate success | **no per-task expected success for SmolVLA** [?] |
| openvla `regenerate_libero_dataset.py` [F] | replays HDF5 demos, keeps those that succeed, writes per-demo metainfo | sim-vs-demo fidelity, not policy wiring; output not published. LIBERO#16: action replay diverges across machines (state distance 0.24–2.04) — maintainer recommends resetting to sim state instead |
| lerobot#4390 `mj_settle_probe.py` [F] | task-5 bowl drift per MuJoCo version, no policy, seconds | one task |
| lerobot PR #4629 `lerobot-eval-compare`, PR #4628 Wilson intervals [S] | per-task regression between two eval outputs | titles only |

A LeRobot maintainer concedes in lerobot#4614 that `lerobot-eval` does not print the effective replan horizon
(`n_action_steps`, `chunk_size`, fps), so two numbers for the same checkpoint are not comparable [F].

### 4.3 What a correct check has to do

1. **Inspect live values, not declared ones.** Read `n_action_steps`, `chunk_size`, `num_vlm_layers`,
   `empty_cameras`, env fps, `control_mode`, observation height/width, `max_parallel_tasks`, and
   `mujoco.__version__` **from the instantiated policy and env at rollout time** — this is the only way to catch
   the config-loss rows (§2.2).
2. **Compare against a hand-curated per-checkpoint reference table** (§5), because `config.json` is not
   authoritative and no machine-readable contract exists anywhere.
3. **Compare shapes, and state/action dimensionality** — closing LeRobot's TODO.
4. **Pixel statistics against the training dataset** — `set_init_state` from a dataset episode, apply the eval
   flip and resolution, diff the rendered first frame against the stored frame (mean pixel gap per suite, as in
   MINERVA EXPERIMENTS.md:500-518). Catches orientation, resolution *and* lighting in one number; CPU-only, cheap.
   **This is the strongest single recommendation.**
5. **Per-scene state check** — first-step `observation.state` vs the dataset's frame-0 state for the same task
   (§2.3 row 17 did this by hand).
6. **MuJoCo version bands** — refuse ≥3.10 (crash), warn ≥3.4 (task-5 physics), warn ≥3.3.3 against standard
   renders (lighting).
7. **Per-task settle probe** for exposure to physics changes, generalising #4390's probe beyond task 5.

`my primary` has implemented 1–3 and 6 in `vla_harness/conformance.py` (MATCH / MISMATCH / UNDECLARED verdicts;
refuses on the original 360/nas10 campaign config). Items 4, 5 and 7 are not yet built.

---

## 5. Q5 — what checkpoint configs actually declare

| Field | `HuggingFaceVLA/smolvla_libero` (6721902b) | `lerobot/pi05_libero_finetuned` | `nvidia/gr00t17-lerobot-libero_spatial-640` | `k1000dai/MINERVA` `t05_l1_0.54M` |
|---|---|---|---|---|
| image keys / shape | `image`, `image2` · [3,256,256] | `image`, `image2` [3,256,256] + `empty_camera_0` [3,224,224] | `image`, `wrist_image` · **[256,256,3] HWC** | `image`, `image2` · [3,256,256] |
| internal image size | `resize_imgs_with_padding` [512,512] | `image_resolution` [224,224] | `image_size` [256,256]; crop 230→256 | resize 144, crop 128 (random) |
| state / action dims | 8 / 7 | 8 / 7 | 8 / 7 | 8 / 7 |
| `chunk_size` / `n_action_steps` in config | 50 / **1** | 50 / **50** | 16 / **16** (preprocessor `action_horizon` 40) | horizon 16 / **8**, `n_obs_steps` 2 |
| **documented eval** | **none found** — card is a generic template; no SmolVLA LIBERO command in LeRobot docs | **`n_action_steps=10`** (`libero.mdx:199-203`) | **none** ("No evaluation results have been provided") | **`n_action_steps=1`, `temporal_ensemble_coeff=0.01`**, 50 eps/task, seed 1000, hard reset, `mujoco==3.3.2`, locked env |
| normalisation | STATE/ACTION MEAN_STD | MEAN_STD | IDENTITY in config; min-max + percentile clip inside `groot_n1_7_pack_inputs_v1` | STATE/ACTION MIN_MAX, VISUAL MEAN_STD |
| training dataset declared | **no** ("datasets: unknown"; stats match `lerobot/libero`) | no (docs: `HuggingFaceVLA/libero`) | `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot` (train_config) | `lerobot/libero` (train_config) |
| flip / camera→key / control mode / render resolution / sim version | **not declared** | **not declared** | **not declared** | sim version in card prose only |

All [F]: local cache for SmolVLA and π0.5; HF `raw/main` files fetched 2026-09-16 for GR00T and MINERVA.

**Findings:**
- **`config.json` `n_action_steps` disagrees with the documented eval for π0.5 (50 vs 10) and MINERVA (8 vs 1 +
  temporal ensembling); SmolVLA and GR00T document no eval at all.** For the latter two the shipped value is a
  default, not a verified eval setting — a gate should report UNDECLARED, not MATCH.
- **GR00T:** camera keys only line up through a hard-coded alias (§4.1); trained on the IPEC-COMMUNITY conversion
  rather than `HuggingFaceVLA/libero`. Whether that dataset shares the 180° orientation `LiberoProcessorStep`
  assumes is **unverified** [?] — a candidate silent failure.
- **SmolVLA:** preprocessor and postprocessor normaliser safetensors are the same blob (`7008ba73…`); the
  preprocessor contains no flip or resize step, so orientation lives entirely in the env-side
  `LiberoProcessorStep` (`processor/env_processor.py:28-75`) and is invisible to the checkpoint.
- LeRobot's `LiberoEnv` *class* default resolution is 256 while its `LiberoEnvConfig` default is 360
  (`envs/configs.py:333-334`) [F].

---

## 6. Checkpoint identity: `HuggingFaceVLA/smolvla_libero` is not the paper's 0.45B model

Three independent strands, all [F]:

1. **Architecture.** `config.json` has `num_vlm_layers: 0`; class default is 16 (`configuration_smolvla.py:96`).
   `smolvlm_with_expert.py:102-105` truncates only if `> 0`; `:112` sizes the expert to match. Parsed
   `model.safetensors` header: **32 VLM text layers and 32 expert layers**; VLM text 361.9M + expert 96.7M + vision
   86.4M + other 59.9M = **604.9M**. Expert depth is fixed at construction, so this was *trained* at full depth —
   not a 16-layer model with dead weights. `expert_width_multiplier: 0.5`.
   The paper's main model uses "only the first 16 layers" and expert width 0.75×d (Q2 agent, arXiv HTML) [F].
   The same conclusion was reached independently in lerobot#2354 [F].
2. **Normalisation provenance.** Stats byte-identical to `lerobot/libero`; the paper trained on
   `physical-intelligence/libero` (§4.1) [F].
3. **No provenance in the artifact.** safetensors `__metadata__` empty; card says "datasets: unknown"; Hub commits
   dated 2025-09-17, after the paper [F].

*Not* evidence (withdrawn): `freeze_vision_encoder=True`, `train_expert_only=True`, `scheduler_decay_steps=30000`
are SmolVLA class defaults (`configuration_smolvla.py:69,70,81`), present in every checkpoint.

**Consequences.**
- **No published LIBERO number exists for this checkpoint.** F8's "61% vs 87%" compares different architecture
  variants.
- The architecturally matched paper rows are the ablation baselines — 32 layers, width 0.5 — which appear across
  Tables 8–13 as 89/94/85/53 (nas=1) and in Table 13 as 89/94/91/57 (nas=10). They were "trained from scratch
  without any pretraining on robotics data" (§4.7) — **architecture-matched but training-mismatched.** They are
  references, not targets.
- **Paired within-checkpoint comparisons are unaffected** (F9's resolution arm; the `n_action_steps=1` arm), since
  both arms share weights.
- Published side is also n=100 per suite (paper §4.1: "10 trials per task") [F].

---

## 7. What this means for our ~20-point residual

### 7.1 Against paper references (n=100 each side; 95% CI on the difference; computed by `vla-81`, spot-checked)

Ours: `res256_nas10_seed1000` — spatial 73.0, object 90.0, goal 68.0, libero_10 37.0 (FINDINGS.md F9 table) [F].

| suite | vs Table 2 headline (90/96/92/71) | vs Table 13 nas=10 ablation (89/94/91/57) |
|---|---|---|
| spatial | 17.0 | 16.0 [5.4, 26.6] z=2.95 |
| object | 6.0 | 4.0 **[−3.5, 11.5] z=1.05 — CI spans zero** |
| goal | 24.0 | 23.0 [12.3, 33.7] z=4.20 |
| libero_10 | 34.0 | 20.0 [6.4, 33.6] z=2.89 |

The large libero_10 gap is mostly an artefact of comparing against the headline model. Caveat: the ablation n per
row is assumed to be 100, not verified [?].

### 7.2 Against community reproductions of the *same checkpoint* — the most like-for-like comparison available

`HuggingFaceVLA/smolvla_libero` via `lerobot-eval`, 10 episodes/task unless noted [F]:

| Source | MuJoCo / nas | spatial | object | goal | long |
|---|---|---|---|---|---|
| lerobot#2354 | 3.3.2 / 1 | 73 | 91 | 83 | 43 |
| lerobot#2354 (Lchaerin) | — | 83 | 96 | 87 | 38 |
| lerobot#2354 (WillMandil001) | — | 82 | — | — | — |
| lerobot#3264 | 3.3.2 / 10 | 63 | 93 | 81 | 56 |
| **ours (res256)** | **3.3.7 / 10** | **73** | **90** | **68** | **37** |

**Spatial, object and long are within the community range; goal is the outlier (68 vs 81/83/87).** Every community
run whose version is stated used MuJoCo 3.3.2; we use 3.3.7. All rows are n=100 per suite, so single-suite
differences under ~10 pp are within noise.

**Independent corroboration from our own mining campaign.** `vla-81` split `runs/camp_20260916-0015_*` by
perturbation arm; nominal episodes (no perturbation, plus `camera_yaw_deg: 0.0`, which skips all camera code)
give object 88.8, goal 75.0, spatial 72.5, libero_10 56.9 (n=80/80/80/65) [as reported by `vla-81`, audit §5c].
Against the repros: spatial in range, libero_10 at or above, object ~4 pp low, **goal the only meaningful gap
(~9 pp)** — same conclusion from a different corpus. The campaign's *perturbed* arms are **not** cited here:
the harness's `camera_yaw` implementation re-aims the camera for any nonzero yaw, adding a ~12° pitch change
that the yaw-0 arm lacks (`vla_harness/envs/libero_env.py:290-313`). **Confirmed by render** (`my primary`): yaw=1e-6 changes the image
as much as yaw=5 (mean abs pixel diff 53.1 vs 52.3 on object, 44.1 vs 45.8 on spatial), so every campaign manifest
row attributing to `camera_yaw_deg` is invalid. Fix pending as `PENDING_DECISIONS.md` #22.

**How noisy is "n=100"?** `my primary`'s paired nas arm (below) had 32 of 100 episodes flip outcome between two
settings with no net effect — 18 succeed only at nas=1, 14 only at nas=10, same seed and init state. Per-episode
churn at this level means our goal gap is 13 pp to the lowest repro (81) and 19 pp to the highest (87): outside
the ~10 pp noise band but not by much. It is a lead, not an established defect, until measured with more episodes
or paired against a single changed variable.

### 7.3 Live candidates, and what's been ruled out

**Ruled out or matching:** `n_action_steps` — Table 13 predicts spatial 89 at both 1 and 10; `my primary`'s paired
arm (`experiments/repro/runs/res256_nas1_seed1000/`, prediction written before the result in `PREDICTION.md`)
measured **77/100 at nas=1 vs 73/100 at nas=10**, +4.0 pp, 95% CI [−8.0, 16.0], exact McNemar p=0.597 [F, as
reported by `my primary`]; `num_steps_wait` (§2.3 row 13), state
frame/normalisation (row 17), instruction strings (row 18), flip (row 14), fps, `max_parallel_tasks`, episode length.

**Live:**
1. **MuJoCo 3.3.7 vs 3.3.2 rendering** — documented as ~15 pp for a pretrained VLA on Object (openvla#282), and the
   one systematic difference between our runs and all community repros. Our own suite profile (object fine, goal
   worst) does not match the Object-heavy lighting signature, which argues against it being the *dominant* cause;
   but it is unmeasured for SmolVLA and is the **cheapest A/B left** (goal suite, 3.3.2 vs 3.3.7, one variable).
2. **Language-conditioned disambiguation** — goal holds one fixed scene and varies only the instruction; it is the
   most language-dependent suite and our largest outlier. `my primary`'s language probe found spatial tasks 0–1 at
   redirect_fraction 0.0 (instruction ignored) vs object tasks 0–2 at 1.0; goal was inconclusive (n=5 seeds, 9
   cells — indicative only). Against it: F9's purely visual resolution fix moved spatial by +17, which a pure
   language account doesn't obviously predict — though resolving "the bowl *between* the plate and the ramekin"
   needs visual detail too, so this does not separate the two.
3. **Geometric imprecision** — competes with (2) for spatial/long; fits goal less well.

**Discriminating experiment for 2 vs 3:** on goal failures, did the policy **achieve a different goal task's BDDL
predicate** (language) or attempt the right goal and miss (geometric)? Goal is ideal because every task shares the
scene. Blocked on data: 200/260 goal traces have `_gt_object_pos_complete = False`, missing exactly the fixture
and region bodies (drawers, stove) that goal predicates reference. `vla-81` is running an indicative pass over the
39 instrumented goal failures. The ~42 min `libero_goal` re-run (`PENDING_DECISIONS.md` #3) is the input that
makes this test decisive, and should verify fixture/region completeness before its traces are trusted.

---

## 8. Open questions, stated as unknowns

- Which MuJoCo version rendered each LIBERO training dataset.
- Whether the IPEC-COMMUNITY LIBERO conversion (GR00T's training data) uses the same 180° orientation convention.
- What produced `HuggingFaceVLA/smolvla_libero` (base, dataset revision, steps) and whether any number was ever
  published for it.
- Whether renderer backend (EGL / OSMesa) affects success.
- Whether robosuite 1.4.1 has the same `mj_fullM` break as 1.4.0.
- The hf-libero fork's diff against upstream LIBERO (not inspected).
- Per-task expected success rates for any SmolVLA checkpoint.

## 9. Sources read in full

LeRobot issues/PRs: #2114, #2354, #3264, #3638, #4315, #4327, #4390, #4465, #4614, #4615, #3319 (CI);
LIBERO #16, #88, #141, PR#140, PR#3 (huggingface/LIBERO); openvla #282; openvla-oft #128, #135, #150;
SimVLA #14; MuJoCo `doc/changelog.rst` (3.3.3, 3.4.0, 3.8.0, 3.9.0) and commit `883836848a67`;
SmolVLA paper arXiv:2506.01844 (HTML: §4.1, §4.3, §4.7, Tables 2, 8, 12, 13); MINERVA HF card and repo
(README, `exp_small_libero/EXPERIMENTS.md`); openpi `examples/libero/{main.py,requirements.txt,Dockerfile,compose.yml}`;
openvla / OFT / VLA-Adapter `experiments/robot/libero/run_libero_eval.py`; GR00T `gr00t/eval/sim/LIBERO/*`;
LIBERO `libero/lifelong/metric.py`, `requirements.txt`, `setup.py`. Local: `third_party/lerobot` (`envs/`,
`policies/factory.py`, `policies/utils.py`, `policies/smolvla/`, `policies/groot/processor_groot.py`,
`processor/env_processor.py`, `docs/source/{libero,pi05,groot,molmoact2}.mdx`), hf-libero / robosuite METADATA,
HF cache for SmolVLA and π0.5 checkpoints and `lerobot/libero`.
