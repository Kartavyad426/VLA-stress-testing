# LIBERO-Plus: perturbation types and difficulty levels

*Written 2026-09-17.* Sources are marked:
- **[paper]** LIBERO-Plus, arXiv:2510.13626 (read via a page summary, not verbatim).
- **[code]** the fork `sylvestf/LIBERO-plus` @ `4976dc3`, which we run.
- **[measured]** computed here from `task_classification.json` with
  `experiments/lplus_level_stats.py`.

Renders of every type at every level for libero_spatial are in
`viz/libero_plus_levels/`, and the full grid is `libero_spatial_all_levels.png`.
Each is one variant, from the same base scene where one existed, showing the
agent-view frame the policy sees.

---

## 0. ⚠ Instruction contamination — read before trusting any LIBERO-Plus result

LIBERO-Plus builds each variant's instruction from its **file name**
(`benchmark/__init__.py:grab_language_from_filename`). For every **non-language**
variant, the name carries the perturbation parameters, and LeRobot passes
`task.language` straight to the policy (`lerobot/envs/libero.py:180`). So the
policy receives, e.g.:

| Variant | Instruction via LeRobot | Correct |
|---|---|---|
| camera 608 | "…place it on the plate **view 0 0 100 2 352 initstate 0**" | "…place it on the plate" |
| noise 1563 | "…on the plate **view 0 0 100 0 0 initstate 0 noise 26**" | "…on the plate" |
| layout 1895 | "…on the plate **level1 sample3**" | "…on the plate" |
| language 988 | "would you mind helping me by picking up…" | same — the rewrite is the perturbation |

**Every non-language LIBERO-Plus variant run through LeRobot perturbs the
language too.**

- **Our harness is fixed (2026-09-17).** `LiberoEnv(libero_plus=True)` strips the
  suffix, so the instruction is the base scene name as words: vanilla LIBERO's
  instruction, which is what the policies were trained on. Language variants keep
  their rewrite. Traces record both `raw_task_language` and `instruction_given`.
- **`lerobot-eval` is NOT fixed.** Its LIBERO-Plus reference runs remain
  contaminated unless LeRobot is patched.
- **Why not the BDDL's own text?** Its `language_instruction` ("pick the akita
  black bowl …") is worded differently from the training data, so it is not the
  right clean text either.
- **Unverified:** whether LIBERO-Plus's published numbers were produced with the
  contaminated or the clean instruction. Their eval scripts are not in this repo.

## 0b. Full variant lists

`docs/libero_plus_variants/<suite>.csv` has one row per variant (10,030 total):
- task_id, category, difficulty level, solved-by-n-of-4
- base scene and decoded parameters
- `instruction_clean` and `task_language_raw`
- variant name

libero_goal has 121 variants with no difficulty level in the upstream catalogue;
they are left blank.

## 1. The single most important fact about levels: level ≠ perturbation strength

**Difficulty level is how many of four reference models solved the variant**, not
how strong the perturbation is. [paper §C.3]

| Level | Meaning |
|---|---|
| L1 | solved by **all four** reference models |
| L2 | solved by exactly three |
| L3 | solved by exactly two |
| L4 | solved by exactly one |
| L5 | solved by **none** |

The reference models are **OpenVLA-OFT, π₀, π₀-fast and UniVLA**.

Consequences:
- **"Level 1" means "easy for those four models", not "mildest perturbation".**
  A level-1 variant can be a strong perturbation those models happened to handle.
- **Levels encode those four models' weaknesses.** A different policy (GR00T,
  SmolVLA, MINERVA) can find a level-5 variant easy and a level-1 variant hard.
  Level is a *prior*, not a ground-truth difficulty for our policies.
- **The released corpus is itself filtered.** The authors removed variants solved
  by all or most models: 10,030 released of 14,000 generated (FINDINGS.md F5). So
  counts per level describe the release, not the perturbation space.
- **Our R-023** (GR00T 53/54 at level 1) therefore tested variants all four
  reference models already solve. A high score there was expected, and is not
  evidence of robustness.

**Measured on libero_spatial, level only loosely tracks strength:**
- **Sensor noise.** Median severity 4 at L1, rising to 9 at L4–L5. But L1 still
  contains severity-8 variants, and in the render grid the L4 variant
  (Gaussian blur) is far blurrier than the L5 one (motion blur).
- **Camera viewpoint.** Median off-axis angle 0° (L1), 23° (L2), **46° (L3)**,
  then 26° (L4) and 22° (L5). The hardest camera variants are *not* the largest
  angles.
- **Lighting.** The L1 render has the strongest colour shift of the five.

---

## 2. The seven perturbation types

| Type | What changes [paper] | How the variant name encodes it [code] | Notes |
|---|---|---|---|
| **Camera Viewpoints** | Distance 1.01–2.00× original; azimuth/elevation within 15–75° cones; camera yaw/pitch/roll 2–10° | `view_H_V_S_R_E`: horizontal angle, vertical angle, distance ×100, two camera-orientation angles | Agent-view camera only |
| **Robot Initial States** | Joint angles perturbed, magnitude 0.1–0.5 | `initstate_N`, N≠0 selects a perturbed robot start | Subtle in renders: small arm pose changes |
| **Language Instructions** | Rewrites: distraction, commonsense replacement, added reasoning complexity | `language_N` indexes a rewrite | Scene unchanged; only the instruction text differs |
| **Light Conditions** | Diffuse colour, light direction, specular intensity, shadows | `light_N` indexes a lighting config | No strength in the name |
| **Background Textures** | Table/scene textures sampled from 950; surface appearance | `table_N` / `tb_N` indexes a texture | No strength in the name |
| **Sensor Noise** | Motion blur, Gaussian blur, zoom blur, fog, glass blur | `noise_N`: 1–10 motion, 11–20 Gaussian, 21–30 zoom, 31–40 fog, 41–50 glass; strength = N within band | **Applied in `step()` to the agent-view image only; the wrist camera is clean** (`env_wrapper.py:288`) |
| **Objects Layout** | Add unseen distractor objects (from 416); perturb the target's pose | `add_N` (N added objects) or `levelK_sampleJ` (target pose shift level K) | Only 6 variants at L1 in libero_spatial |

---

## 3. How many variants per type and level — libero_spatial [measured]

| Type | L1 | L2 | L3 | L4 | L5 | Total |
|---|---|---|---|---|---|---|
| Background Textures | 99 | 87 | 55 | 16 | 1 | 258 |
| Camera Viewpoints | 67 | 133 | 114 | 40 | 22 | 376 |
| Language Instructions | 115 | 100 | 58 | 82 | 35 | 390 |
| Light Conditions | 52 | 104 | 92 | 38 | 6 | 292 |
| Objects Layout | 6 | 40 | 141 | 153 | 45 | 385 |
| Robot Initial States | 90 | 69 | 75 | 46 | 70 | 350 |
| Sensor Noise | 51 | 136 | 95 | 57 | 12 | 351 |
| **Total** | 480 | 669 | 630 | 432 | 191 | **2,402** |

Suite sizes: libero_spatial 2,402 · libero_object 2,518 · libero_goal 2,591 ·
libero_10 2,519. For another suite, run
`python3 experiments/lplus_level_stats.py <suite>`.

Very uneven cells limit what a balanced design can use:
- **Background L5 has 1 variant and Light L5 has 6**, so a "level 5, every type"
  run is not balanced.
- **Objects Layout L1 has 6.**

---

## 4. Inside the name: what each level contains [measured, libero_spatial]

**Camera Viewpoints.** The off-axis angle is approximate: max of the azimuth and
elevation offset.

| Level | Off-axis angle, median (range) | Distance, median (range) |
|---|---|---|
| L1 | 0° (0–58) | 100% (100–168) |
| L2 | 23° (0–75) | 100% (100–199) |
| L3 | 46° (0–75) | 100% (100–200) |
| L4 | 26° (0–73) | 100% (100–197) |
| L5 | 22° (0–71) | 100% (100–175) |

**Sensor Noise**

| Level | Noise types (count) | Severity within band, median (range) |
|---|---|---|
| L1 | zoom 19, fog 16, motion 6, glass 6, gaussian 4 | 4 (1–8) |
| L2 | glass 38, motion 37, gaussian 27, fog 18, zoom 16 | 5 (1–10) |
| L3 | glass 35, gaussian 25, motion 22, zoom 8, fog 5 | 8 (3–10) |
| L4 | gaussian 20, fog 12, zoom 10, motion 9, glass 6 | 9 (4–10) |
| L5 | gaussian 6, zoom 3, motion 2, fog 1 | 9 (7–10) |

**Objects Layout**

| Level | Added-object variants (median n added) | Target-pose-shift variants (shift levels) |
|---|---|---|
| L1 | 1 (8) | 5 (1–2) |
| L2 | 21 (16) | 19 (1–5) |
| L3 | 59 (14) | 82 (1–5) |
| L4 | 99 (19) | 54 (1–5) |
| L5 | 28 (24) | 17 (2–5) |

---

## 5. How to use levels in this project

1. **Treat level as a prior about four other models, not a difficulty scale for
   ours.** Report results per perturbation type *and* level, and never pool
   levels as if they were increasing strength.
2. **For sensitivity questions, use the decoded parameters**, not level.
   - camera: off-axis angle and distance
   - sensor noise: noise type and severity
   - layout: number of added objects and pose-shift level

   These are actual strengths of the perturbation.
3. **To find failures quickly, sample L4–L5.** Those are the variants the
   reference models mostly failed.
4. **Balance around sparse cells** (Background L5 = 1, Light L5 = 6, Layout L1 = 6).
5. **Running:**
   - Use `.venvs/libero-plus`, with `PYTHONPATH=third_party/LIBERO-plus` and
     `LIBERO_CONFIG_PATH=third_party/libero-plus-config`.
   - Through our harness: `experiments/harness_eval.py --libero-plus`. Each trace
     records its category, level and catalogue id.
   - MuJoCo stays at **3.3.2**. LeRobot's LIBERO-Plus Dockerfile pins 3.7.0,
     inside the range that corrupts libero_spatial task 5.

---

## 6. Unverified

- The per-type parameter ranges in §2 come from a summary of the paper, not a
  verbatim read.
- Whether the four reference models were evaluated at a fixed seed / episode
  count when levels were assigned (it affects how noisy the labels are).
- Level labels for the other three suites have not been decoded here; run the
  stats script.
