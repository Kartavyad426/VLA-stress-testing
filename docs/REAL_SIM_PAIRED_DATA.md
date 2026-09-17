# REAL–SIM PAIRED DATA

*Written 2026-09-17. Datasets containing both real and simulated robot video, ideally of the same
task, scene or episode. For (a) validating that sim-mined failures predict real behaviour,
(b) calibrating the sim–real gap, (c) building scenes matched to real footage.*

> Marks: **[F]** full text / artefact read · **[A]** abstract or landing page · **[S]** search-snippet
> · **[?]** unverified. Peer-reviewed, preprint and demo are distinguished throughout.
> Companion: `docs/SIM_TO_REAL_TRANSFER.md`, `docs/DATA_AND_BENCHMARKS.md`.

---

## 0. The three direct answers

**1. Is there any dataset with the same POLICY run in matched real and sim scenes, with per-episode
outcomes released?**

**Yes for the evaluation, [?] for the per-episode release.** **PolaRiS**
([arXiv:2512.16881](https://arxiv.org/abs/2512.16881), **preprint**, 2025-12) is the direct hit:
**6 paired real-and-simulated environments across two institutions**, **four VLAs run in both
domains** (π₀, π₀-FAST, PaliGemma-binning, π₀.₅), **20 real rollouts per policy per environment** and
**50 sim rollouts per task**, with real rollouts human-graded against *"the same rubric used in
simulation"* **[A]**. Reported **Pearson r = 0.9 average, worst environment 0.81, and r = 0.98
against RoboArena** **[A]**.

**SIMPLER** (CoRL 2024, peer-reviewed) qualifies at the *results* level — its Tables IV–VI already
publish real and sim success side by side per policy per task, which is where
`SIM_TO_REAL_TRANSFER.md` §1.1 got its evidence. Whether the underlying real rollouts are released as
data is **[?]**.

**2. Is there any dataset pairing real footage with a reconstructed sim scene of that same footage?**

**Yes, several — but they release *assets*, not paired video.**
`nepfaff/scalable-real2sim` (**71 GB, MIT**) ships `object_data/` and `robot_system_id_data/` for
physics-aware asset generation **[A]**. **Real2Render2Real** (CoRL 2025, peer-reviewed) reconstructs
from a phone scan plus one human demo **[A]**. **RoboMIND 2.0**
([arXiv:2512.24653](https://arxiv.org/abs/2512.24653)) is the largest: it *"additionally provides
simulation assets of all objects used in the dataset, along with their integration into Isaac Sim"*,
plus **20K simulated trajectories on the same tasks as the real data** **[S]**.

**3. Which would I pick first?** — **None of them, for the reason in §4.** The honest answer is that
the paired data that would validate *our* pipeline does not exist and cannot be downloaded; what these
datasets validate is *their* simulators. PolaRiS is the one to copy — **its protocol, not its data.**

---

## 1. "Paired" means four different things, and the brief needs three of them

This distinction does most of the work in this document, because the word hides a 100× difference in
how hard the data is to get.

| type | what it means | who has it | serves |
|---|---|---|---|
| **1. Same episode** | the *same trajectory* executed or replayed in both domains | **essentially nobody** (§5) | the strongest form of (b) |
| **2. Same scene + same policy** | independent rollouts of one policy in a real scene and its digital twin | **PolaRiS**, SIMPLER | **(a)** and (b) |
| **3. Same task, different scenes** | sim data of the same task, not the same room | RoboMIND 2.0, co-training work | training, **not** validation |
| **4. Real footage → reconstructed asset** | a scene rebuilt from real capture; no paired rollouts | scalable-real2sim, R2R2R, ManiSkill real2sim | **(c)** |

> **Use (a) — *do sim-mined failures predict real behaviour* — requires type 2 and nothing less.**
> Type 3 tells you a policy trained on both does better; it says nothing about whether a failure found
> in sim is a failure in reality. Several datasets advertise "real and sim" and are type 3.

---

## 2. The datasets

### 2.1 PolaRiS — the only true type-2 release found

*[arXiv:2512.16881](https://arxiv.org/abs/2512.16881) · preprint · code `arhanjain/PolaRiS` · assets
`owhan/PolaRiS-Hub` on HF · site polaris-evals.github.io*

| field | |
|---|---|
| **what is paired** | **same scene + same task**, real and simulated; independent rollouts, not the same episode |
| **scale** | 6 paired evaluation environments, 2 institutions; 6 tasks (Food Bussing, Block Stacking, Pan Cleaning, Move Latte Cup, Organize Tools, Tape Into Container); a further 15 real-to-sim scenes for co-training collection |
| **policies in both domains** | **π₀, π₀-FAST, PaliGemma-binning, π₀.₅** — 4 VLAs |
| **rollouts** | real **20 per policy per environment**; sim **50 per task** |
| **correlation** | Pearson **r = 0.9** mean, **0.81** worst environment, **0.98** vs RoboArena |
| **simulator** | **IsaacSim** |
| **privileged state** | **[?]** — IsaacSim has it; reachability through their wrapper unchecked |
| **licence / size** | **[?]** — not stated on the site |
| **per-episode outcomes released** | **[?]** — the site advertises environments and sample videos; granular outcome data is not confirmed |
| **our stack** | ✗ IsaacSim, and `DATA_AND_BENCHMARKS.md` already rates Isaac as implausible on 8 GB |

**Verdict:** (a) **the best available template**, (b) good, (c) no. The single most useful thing in it
is the *protocol* — paired environments, a shared rubric across domains, 20 real rollouts per cell.

### 2.2 SIMPLER — paired *results*, uncertain paired *data*

The real-vs-sim tables are the evidence base already used in `SIM_TO_REAL_TRANSFER.md` §1.1, including
**Table VI**, the only published per-*perturbation-condition* pairing anyone has. `ManiSkill_bridge_v2_real2sim`
(HF, **MIT, 82 MB**) ships *"Assets for Real2Sim evaluation of the Bridge v2 dataset"* **[F]** — the
Bridge digital twin, **assets only, no video**.

**Verdict:** we have already extracted its value by reading the tables. The asset repo helps only if
we adopt SAPIEN.

### 2.3 RoboMIND 2.0 — the largest same-task real+sim corpus

**20K simulated trajectories in Isaac Sim** on dual-arm Franka and Tien Kung, *"performing the same
tasks as in the real-world dataset"*, plus **simulation assets of all objects in the dataset** **[S]**.
Reports that hybrid real+sim training beats real-only, *"validating the high fidelity of the digital
twin"*.

**Verdict:** **type 3, not type 2.** Excellent for co-training evidence and for (c) via the asset
release; **does not answer (a)**, because same-task is not same-scene and no policy is evaluated in
both domains under a shared rubric. Also the same corpus whose failure subset we examined in
`DATA_AND_BENCHMARKS.md` §6.

### 2.4 Real-to-sim asset pipelines — for use (c)

| | |
|---|---|
| **`nepfaff/scalable-real2sim`** | **71 GB, MIT** **[A]**. `object_data/` (per-object TAR archives) + `robot_system_id_data/`; physics-aware asset generation, i.e. it estimates physical parameters rather than appearance alone. **The most relevant single artefact for (c).** Target simulator not stated **[?]** |
| **Real2Render2Real** ([arXiv:2505.09601](https://arxiv.org/abs/2505.09601), **CoRL 2025, peer-reviewed**) | phone scan + **one** human demo → thousands of rendered demos via 3DGS, meshed for IsaacLab. Claims parity with **150 teleop demos**, at 1/27 the time. Converted datasets released for diffusion policy and π₀-FAST **[A]** |

> **R2R2R corroborates the conclusion reached in this session independently**, and it is worth
> recording because it is the mechanism rather than an opinion: R2R2R converts its splat
> representations to meshes for the renderer **with collision modelling switched off** **[A]**. A
> pipeline built by people who do this for a living turns physics *off* when reconstructing from
> footage. **Splatting gives appearance; physics needs object separation plus assumed mass and
> friction, and nobody gets it free from video.**

### 2.5 Also checked

- **`RobotControlStack/rcs_real_sim_async_all_500`** — **Apache 2.0**, 150 episodes / 168K rows,
  dual-camera 256×256, joint states and actions, **Parquet, not LeRobot** **[A]**. Real and sim
  episodes in one repo, but "**async**" and no evidence of scene-level pairing → **type 3 at best**,
  possibly just both domains unrelated. Small.
- **RoboDojo** ([arXiv:2607.04434](https://arxiv.org/abs/2607.04434), preprint) — **42 simulation
  tasks and 18 real-world tasks**, five evaluation dimensions, Isaac Sim, plus *RoboDojo-RealEval*, a
  **remote cloud real-robot evaluation service** with standardised hardware and reset. 30 policies on
  a public leaderboard **[S]**. The sim and real task sets are **complementary, not matched**, so it
  is type 3 — **but the remote real-eval service is independently interesting** as a way for a
  one-arm team to get real numbers without owning the cell.
- **RoboManipBaselines** ([arXiv:2509.17057](https://arxiv.org/abs/2509.17057)) — unified imitation
  framework across real and sim **[S]**; framework, not paired data.
- **AutoEval** ([arXiv:2503.24278](https://arxiv.org/abs/2503.24278)) — autonomous *real-world*
  evaluation; a substitute for sim evaluation rather than a pairing.

---

## 3. Comparison

| dataset | pairing type | real video | sim video | privileged state | policies in both | our stack | licence |
|---|---|---|---|---|---|---|---|
| **PolaRiS** | **2 — scene+policy** | ✅ | ✅ | **[?]** | **4 VLAs** | ✗ IsaacSim | **[?]** |
| SIMPLER | **2 (results)** | **[?]** | ✅ | ✅ SAPIEN | 6 (Google) / 3 (WidowX) | ✗ SAPIEN | MIT (assets) |
| RoboMIND 2.0 | 3 — same task | ✅ | ✅ 20K traj | Isaac | none | ✗ IsaacSim | **[?]** |
| scalable-real2sim | 4 — assets | ✅ | asset only | n/a | none | **[?]** | MIT |
| Real2Render2Real | 4 — assets | scan+1 demo | ✅ rendered | collision **off** | none | ✗ IsaacLab | **[?]** |
| ManiSkill bridge real2sim | 4 — assets | ✗ | asset only | n/a | none | ✗ SAPIEN | MIT |
| rcs_real_sim_async_500 | 3 or none | ✅ | ✅ | ✗ | **[?]** | Parquet | Apache 2.0 |

---

## 4. Which to pick — and why the answer is "none, copy the protocol"

**Every type-2 dataset runs on a simulator we cannot run**, and that is not a coincidence — paired
real-sim evaluation is built by labs with real robots and Isaac-class compute. PolaRiS is IsaacSim,
SIMPLER is SAPIEN, RoboMIND 2.0 is IsaacSim. None is MuJoCo or robosuite, so none carries our
privileged-state mining across.

**But the deeper reason not to pick one is a category error worth naming.** These datasets establish
that **their** simulator predicts **their** real setup. Using PolaRiS's r = 0.9 to argue that *our*
LIBERO findings transfer would be borrowing someone else's validation for our own instrument. The
correlation is a property of the *pair*, not of simulation in general — which is exactly the premise
the request started from: **realism comes from matching a specific real scene**, so a correlation
measured on their scenes says nothing about ours.

**So the recommendation is:**

1. **Copy PolaRiS's protocol.** Paired environment, identical rubric scored across both domains, ~20
   real rollouts per cell. That is a design we can execute at small scale and it is the same shape as
   the cheapest-real-check already proposed in `SIM_TO_REAL_TRANSFER.md` §4 — three factors, ~20
   trials per factor per level, compare *ordering*. **The two recommendations are the same
   experiment**, arrived at from different directions, which is mild evidence it is the right one.
2. **If use (c) is pursued — building scenes from real footage — start with
   `nepfaff/scalable-real2sim`**: MIT, and the only artefact found that ships *physical-parameter*
   estimation rather than appearance alone. That is precisely the gap this session identified.
3. **Watch RoboDojo-RealEval.** A remote real-robot evaluation service with standardised hardware is
   the cheapest conceivable route to real numbers for a team with one arm, and possibly cheaper than
   using the arm.

---

## 5. What does not exist

Stated plainly, because these are the gaps that would otherwise be assumed filled.

- **Type-1 pairing — the same episode in both domains — is essentially unavailable.** Nothing found
  replays a recorded real trajectory through a reconstructed sim of that same scene and releases both.
  The closest relative is the 2019 benchmark *Benchmarking Simulated Robotic Manipulation through a
  Real World Dataset* ([arXiv:1911.01557](https://arxiv.org/abs/1911.01557)), which distributes a real
  ground-truth dataset and asks users to replicate its results in simulation **[S]** — the right idea,
  seven years old, and not VLA-era.
- **No paired real-sim dataset exists on MuJoCo or robosuite**, so nothing here carries our mining
  layer across without a new simulator stack.
- **No dataset pairs real and sim *failure* outcomes**, which is what would actually validate failure
  mining rather than success-rate correlation. `SIM_TO_REAL_TRANSFER.md` §1.2 reached the same
  conclusion from the methods side; this is the data side of the same hole.
- **No released per-episode outcome table** confirmed for any type-2 dataset **[?]** — the
  correlations are published, the per-episode data behind them may not be.

---

## 6. Unverified

| claim | mark | to settle |
|---|---|---|
| PolaRiS licence, size, per-episode outcome release | **[?]** | read `arhanjain/PolaRiS` and `owhan/PolaRiS-Hub` |
| PolaRiS privileged-state reachability through its wrapper | **[?]** | read the env code |
| Whether SIMPLER releases its *real* rollouts as data | **[?]** | check the repo's data links |
| RoboMIND 2.0 sim-data availability and licence | **[S]** | the 2512.24653 PDF exceeded fetch limits; not read at full text |
| `scalable-real2sim` target simulator | **[?]** | read the repo |
| `rcs_real_sim_async_all_500` — what "async" pairs, if anything | **[?]** | read the card and one episode |
| RoboDojo-RealEval access model and cost | **[S]** | read robodojo-benchmark.com |
| Whether a `droid_sim` real counterpart exists | **[?]** | carried over unresolved from `DATA_AND_BENCHMARKS.md` §5 |
