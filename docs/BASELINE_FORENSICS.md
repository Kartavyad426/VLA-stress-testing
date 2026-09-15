# Baseline forensics — why is our SmolVLA number 26 points low?

**Written 2026-09-15**, after run `nas10_seed1000`: **61.3%** overall against a
published ~87.3%, with published outside the 95% CI on all four suites (F8).

---

## 0. It is not our harness, and that is established by construction

**`vla_harness` was not involved in this run.** It went through `lerobot-eval`
as shipped, from a pinned source checkout. Our code contributed the wrapper
script, the provenance block, and nothing that touches the policy, the
environment, the rollout loop or the success predicate.

That was the reason for using the standard tool for this phase
(`EXPERIMENT_PROCEDURE.md` §0), and it has paid off: a whole class of
explanation is excluded without further work.

So the difference is in **configuration, environment or the checkpoint itself**.

---

## 1. THE LEAD: we render at 360×360; the checkpoint was trained on 256×256

Highest-confidence discrepancy found, and it is cheap to test.

| | Value | Source |
|---|---|---|
| `LiberoEnvConfig.observation_height/width` | **360 × 360** | `envs/configs.py:333` — **this is what `lerobot-eval` uses** |
| `LiberoEnv` class default | 256 × 256 | `envs/libero.py:119` — *not* the path taken |
| Checkpoint `input_features` shape | **256 × 256** | `smolvla_libero/config.json` |
| Training dataset `lerobot/libero` | **256 × 256** | dataset `info.json` |
| Confirmed in our run | `'observation_height': 360` | `baseline.log` |

**The policy then resizes with padding to 512×512** (`resize_imgs_with_padding:
[512, 512]`). So:

```
TRAINING:   render 256  ->  resize x2.00 -> 512  -> VLM
OUR EVAL:   render 360  ->  resize x1.42 -> 512  -> VLM
```

**A 2.00× upscale and a 1.42× resample do not produce the same image.** The
first is a clean integer upscale; the second introduces interpolation blur and
aliasing, and changes the high-frequency content a SigLIP encoder is sensitive
to. The policy sees a different image distribution from the one it was
fine-tuned on, on **every frame of every episode**.

**Why it is plausible as the main cause:** it affects all four suites (it does),
it is a *perception-side* shift (and our worst suites are the ones where visual
precision matters most), and nothing about it would show up as an error.

**Test:** re-run with `--env.observation_width=256 --env.observation_height=256`.
**~2 h, and it is the cheapest test we have.** It should be run before the 9-hour
`n_action_steps` experiment.

> **Not proven.** LeRobot's default is presumably what their own reproductions
> use, so if 360 were fatal one might expect it to be more widely reported. But
> LeRobot's documented reproduction is of **π0.5**, not SmolVLA — and π0.5 may
> be less resolution-sensitive, or trained at a different resolution.

---

## 2. Full configuration comparison

Confirmed from `baseline.log` unless noted.

| Parameter | Ours | Published / expected | Match? |
|---|---|---|---|
| **render resolution** | **360×360** | **256×256** | **❌ see §1** |
| `n_action_steps` | **10** | **unknown** — shipped config says **1** | **⚠ §3** |
| `chunk_size` | 50 | 50 | ✅ |
| `n_obs_steps` | 1 | 1 | ✅ |
| `num_steps` (flow integration) | 10 | 10 | ✅ |
| `control_mode` | relative | relative (LeRobot default) | ✅ |
| `init_states` | True | True | ✅ |
| `hard_reset` | True | True (required for reproduction) | ✅ |
| `resize_imgs_with_padding` | (512,512) | (512,512) | ✅ |
| `tokenizer_max_length` | 48 | 48 | ✅ |
| `max_state_dim` / `max_action_dim` | 32 / 32 | 32 / 32 | ✅ |
| `train_expert_only` | True | True | ✅ |
| `use_amp` | False | unknown | ⚠ |
| **episodes/task** | **10** | **unknown; 10–50 both cited** | ⚠ §4 |
| **seeds** | **1** (seed 1000) | LeRobot recommends **3** | ⚠ §4 |
| mujoco | **3.3.7 (healthy)** | 3.4.0–3.8.1 are broken (F2) | ✅ better |
| LeRobot | `b6ec006` | unknown | ⚠ |
| dataset revision | `a1aaacb7…` (pinned) | unknown | ⚠ |
| GPU | RTX PRO 1000 Blackwell, 8 GB | unknown, likely A100-class | ⚠ §5 |

---

## 3. `n_action_steps` — the second candidate

The checkpoint ships **`n_action_steps: 1`** against `chunk_size: 50`. We ran
**10**, matching LeRobot's π0.5 reproduction, which passes
`--policy.n_action_steps=10` explicitly and notes it matches OpenPI.

**Nobody documents what SmolVLA's published numbers used.** This changes the
open-loop horizon and therefore behaviour: at 1 the policy re-plans every step;
at 10 it commits to 10 actions before looking again.

Cost to test: 3.95 vs 17 steps/s ⇒ **~9 h**. Do §1 first.

---

## 4. Sampling — cannot explain a 26-point gap on its own

10 episodes/task with one seed. LeRobot recommends averaging 3 seeds, and the
survey notes 10 vs 50 can swing spatial/long by 15–20 pp.

**But this is variance, not bias.** It cannot systematically depress all four
suites in the same direction. It widens our intervals; it does not move the
centre. Published sits outside the CI on every suite.

---

## 5. Hardware — the least likely explanation

8 GB Blackwell laptop GPU versus, presumably, datacentre hardware. Different
kernels and reduction orders ⇒ different floats ⇒ **contact chaos can flip
individual episodes** (`ARCHITECTURE.md` §5.1).

**But that is noise, not bias**, and it would have to be implausibly large and
one-directional to account for 26 points. Precision is a more plausible route
(`use_amp: False` here — unknown elsewhere), and worth checking, but it is not
where I would look first.

---

## 6. What the evidence rules OUT

- **Our harness** — not in the loop (§0)
- **The MuJoCo physics bug (F2)** — we ran healthy 3.3.7. Reporters on broken
  versions should have scored *worse* than us, and scored **better**
- **Multiplicity** — 18.5% predicts one spurious flag across four suites; all
  four are flagged
- **A single broken suite** — all four are down, by 11 to 36 points

---

## 7. What makes this hard to dismiss

**We are the lowest of three independent reports.** #3264 got 73.25%, #3287
~67%, we got 61.3%. Three parties, three numbers, all far below published, none
reproducing.

**`libero_10` shows the largest gap (−36).** The long-horizon suite is where
compounding error predicts most damage, and where language reportedly matters
most even for models that ignore it elsewhere.

**`libero_object` is closest (−11)** — the suite the literature calls least
ambiguous, where the scene most strongly identifies the task.

---

## 8. Ranked next tests

| # | Test | Cost | Why |
|---|---|---|---|
| **1** | **Re-run at 256×256** | **~2 h** | **Concrete documented mismatch, affects every frame, cheapest test** |
| 2 | Re-run at `n_action_steps=1` | ~9 h | Shipped default; changes open-loop horizon |
| 3 | π0.5 anchor on a ≥16 GB card | rented | **The only test that separates "our setup" from "SmolVLA does not reproduce"** |
| 4 | 3 seeds × 50 eps/task | 15–24 h | Tightens intervals; will not move the centre |
| 5 | Switch subject to VLA-Adapter | ~2 d | Its known gap has an identified cause |

**Tests 1 and 2 are confounded with each other and must be run separately**, one
variable at a time — which is the same discipline the counterfactual probe
applies to perturbations, and for the same reason.
