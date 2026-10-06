# VLA harness specifics (repo `~/Documents/Code/VLA`)

Read this when the post-mortem is about a run in this repo. The worked example, R-056/R-058 (GR00T N1.7 start-pose LoRA pilot), is in `docs/R056_POSTMORTEM.html`. All commands run from the repo root.

## House rules
- **Analysis is CPU-only, from existing artifacts.** Scripts go in a scratch folder or `experiments/analysis_<topic>/`. Don't edit the pipeline code.
- **GPU work.** Any new GPU run must first be pre-registered in `RESULTS.md` and approved by the user, through the SFT session when it commissioned the work. Hold `flock /tmp/vla_gpu.lock`, announce to "my prim", and check the laptop is on AC power.
- **Commits.** Don't commit unless the user asks.
- **Deliverables.** Write standalone HTML in `docs/`. Never publish an Artifact.
- **Reporting back.** If a peer session (e.g. SFT) asked for the analysis, reply with SendMessage to its `from` address, giving numbers, paths and the data/hypothesis split.
- **Subagents.** Don't spawn subagents here; work directly.

## Where things are
| what | path |
|---|---|
| pre-registrations, amendments, notes | `RESULTS.md` (search `## R-0NN`) |
| overnight headlines / queue log | `runs/overnight/SUMMARY_*.txt`, `runs/overnight/queue.log` |
| training logs (lerobot_train stdout) | `runs/overnight/<RUN>_train.log` |
| training report (parts, shares, effective batch, LoRA, augmentation state, LR trace) | `runs/<run>_train_report.json` |
| checkpoints (dir names = MICRO-steps; optimizer step in `vla_train_meta.json`) | `runs/<run>_train/checkpoints/<micro>/pretrained_model/` |
| eval run: manifest, rollouts, score, selection, transfer | `runs/<run>/{manifest,rollouts}.jsonl`, `score.json`, `selection.json`, `transfer/` |
| minting run (all attempts, incl. failures) | `runs/r054/{manifest,rollouts}.jsonl`, `episodes/*.json`, `yield.json` |
| exported training set (LeRobot v3) + provenance | `data/<name>/data/*/*.parquet`, `data/<name>/meta/{provenance.jsonl,export.json,tasks.parquet}` |
| replay set (IPEC libero_spatial_no_noops, v3) | `~/.cache/huggingface/lerobot/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot/` |
| base checkpoint's original recipe | `~/.cache/huggingface/hub/models--nvidia--gr00t17-lerobot-libero_spatial-640/snapshots/*/train_config.json` |
| nominal (unperturbed) start per scene | `runs/r047/nominal_starts.json` |
| held-out / validation start lists | `experiments/repro/r056_{heldout,val}.json` |
| scorer / trainer / exporter | `experiments/r056_eval.py`, `experiments/r056_train.py`, `vla_harness/data/export_lerobot.py`, `vla_harness/training/{mix,groot_lora,schedule}.py` |

## Venvs
- `.venvs/groot/bin/python`: numpy and pandas analysis (`paired`, `data_coverage`, `grasp_events`, `recipe_diff`).
- `.venvs/libero-plus/bin/python`: anything that imports lerobot training code (`padcheck`); it is the training venv.
- System `python3`: has no numpy. Only `sample_accounting.py` and `loss_trace.py` run under it.

## Phase → script
| phase | command |
|---|---|
| 2, 3, 4: paired table, leave-one-group-out, offset split with composition, failure mode, first-chunk shift, nominal failures | `.venvs/groot/bin/python .claude/skills/finetune-postmortem/scripts/paired.py --run runs/r056_r16=r16 --run runs/r058_r4=r4 --nominal-starts runs/r047/nominal_starts.json` |
| 4: close geometry per scene × arm × outcome | `... scripts/grasp_events.py eval --run runs/r056_r16=r16 --run runs/r058_r4=r4 --scenes 1327` |
| 5: coverage and gaps (scenes, offset bins and yield, nearest demo, window phases, replay start spread, marginals with window-matched control) | `... scripts/data_coverage.py --mint-run runs/r054 --export data/r054_train48 --replay ~/.cache/huggingface/lerobot/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --eval-run runs/r056_r16 --arm-run runs/r056_r16=r16 --arm-run runs/r058_r4=r4 --nominal-starts runs/r047/nominal_starts.json` |
| 6: failed-first-grasp demos, defect inside window, recovery cut | `... scripts/grasp_events.py demos --mint-run runs/r054 --export data/r054_train48` |
| 7: dose (views per frame at each checkpoint) | `python3 .claude/skills/finetune-postmortem/scripts/sample_accounting.py --report runs/r056_r16_train_report.json --checkpoints 500,1000,2000` |
| 8: recipe diff | `.venvs/groot/bin/python .claude/skills/finetune-postmortem/scripts/recipe_diff.py --base nvidia/gr00t17-lerobot-libero_spatial-640 --finetune runs/<run>_train/checkpoints/last/pretrained_model --report runs/<run>_train_report.json` |
| 8: loss trace | `python3 .claude/skills/finetune-postmortem/scripts/loss_trace.py runs/overnight/<RUN>_train.log --accum 8` |
| 10: pad-mask rule-out | `.venvs/libero-plus/bin/python .claude/skills/finetune-postmortem/scripts/padcheck.py --root data/<name> --repo local/<name>` |

Phase 9 (selection) is read from `runs/<run>/selection.json`: the validation n per checkpoint, and whether a `base` row exists on the val set (count the manifest rows with ckpt=base and set=val).

## Known gotchas (each cost time once)
- **`rollout_id` ignores the noise seed.** In one run it had 172 unique values for 316 rows. Join `manifest.jsonl` to `rollouts.jsonl` **by position**; the scripts assert `rollout_id` and `success` match row by row. Never build a dict keyed on `rollout_id`.
- **Radius labels are requested, not achieved.** The achieved joint radius is about 0.74–0.78 of the requested value. Gripper offset in cm from `nominal_starts.json` is the physical axis, and it varies by direction (8 cm vs 17 cm median at the same radius in R-056).
- **Held-out uses only 2 directions per scene**, so direction is confounded with offset, and success depends on which directions were drawn.
- **Target object.** It is the first key of `steps[0].obs_state._gt_object_pos` (the bowl in libero_spatial). A lift means a z rise of more than 3 cm.
- **Gripper conventions.** Env actions use −1 = open, +1 = close; the dataset uses 1 = open, 0 = close (export writes (1−g)/2). A close event is the executed action crossing from ≤ 0 to > 0.
- **Training-log format.** lerobot logs every `log_freq` = 10 MICRO-steps, and each `loss:` is a 10-sample mean. Steps ≥ 1000 print as "1.2K", so parse in file order. The tqdm `\r` must be normalised.
- **The report's `curve.train` is empty** unless `--eval-every` was set, so the log is the only loss trace. Per-source loss is not recoverable from it; reconstructing the sampler stream gave R² ≈ 0. Use a fixed-seed per-subset loss eval on GPU instead.
- **Mixed dataset.** `MixedDataset.meta` is the FIRST part's metadata. Task strings still resolve per part, even though the parts' `task_index` tables differ in order; `padcheck.py` prints the task.
- **Augmentation.** It is off under `lerobot_train` (processors `training=False`), and the checkpoint's own recipe used ColorJitter with state dropout 0.2. The finetune `train_config.json` shows `state_dropout_prob: null`; the processor's actual state is in the run report's `augmentation`.
- **Base checkpoint recipe.** Batch 320, 20k steps, LR 1e-4, 1,000 warmup steps, tunes the DiT, projector and VLLN.
- **Selection.** Base is not run on the validation set, and only the selected checkpoint is run on held-out.
- **Sim render nondeterminism.** A plain re-run flips about 30–40% of failures (R-048, R-053). Read discordance counts against that.
