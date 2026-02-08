# Experiment: Training Improvements for Chunk Continuity

## Research Goal

Improve chunk boundary smoothness through training-only modifications (no inference-time tricks like temporal ensembling / overlap blending).

**Prior work** (from `2026-02-07_22-05-improve-chunk-continuity`):
- Lower LR fine-tuning (lr=3e-6) was the only training modification that helped: U_a 0.134 vs 0.263 baseline
- Auxiliary losses (boundary loss, accel penalty, front-weighted loss) all made things worse
- Overlap blending (inference trick) gave U_a 0.058, but we want training-only improvements

## Experiment Sequence

### 01 -- LR Sweep (5K steps from scratch)

**Machine 1.** Quick sweep to find the best learning rate for from-scratch training. LR values: [1e-5, 3e-5, 5e-5, 1e-4, 3e-4]. 5K steps each. Pick the LR with lowest training loss (we expect a sweet spot).

### 02 -- Observation Dropout (30K steps from scratch)

**Machine 1.** After LR sweep completes.

Stochastically drop proprioception (observation.state) during training. This forces the model to rely more on the action prefix for predicting continuations, rather than memorizing observation->action mappings. Images are NOT dropped.

Implementation: During training, with probability `p_drop_obs`:
- Replace proprioception (observation.state) with zeros
- Images and prefix tokens remain intact

The model must learn: "even if I can't see my joint state, I can still predict the next chunk by extending the prefix trajectory smoothly."

Sweeps `p_drop_obs` over {0.1, 0.3, 0.5} (~2hrs per value, ~6hrs total).

### 03 -- Relative Action Prediction (30K steps from scratch)

**Machine 2.** After LR sweep determines best LR.

Predict action chunks as deltas relative to `a_{t_0}` (the action at the observation timestep, always in the prefix). This architecturally enforces C0 continuity: predictions are anchored to the known action.

Implementation:
- All actions (prefix + targets) are transformed: `a'_i = a_i - a_{t_0}`
- Prefix becomes: `[a_{-K} - a_0, ..., a_{-1} - a_0, 0, a_1 - a_0, ...]`
- Target becomes: `[a_D - a_0, ..., a_{D+C-1} - a_0]`
- Model predicts deltas, output is converted back: `pred_abs_i = pred_delta_i + a_{t_0}`
- At inference, same transform/inverse applied

## Machines

| Machine | SSH | GPU |
|---------|-----|-----|
| 1 | `ssh giacomoran@213.192.2.117 -p 40048` | RTX 3090 |
| 2 | `ssh giacomoran@213.192.2.92 -p 40169` | RTX 3090 |

## Evaluation

Same offline replay eval as before:
```bash
python scripts/eval/eval_offline_replay.py \
    --path_policy=<checkpoint>/pretrained_model \
    --id_repo_dataset=giacomoran/lerobot_policy_act_smooth_30fps \
    --indices_episode=0,1,2 \
    --path_output=<output_path>
```

## Machine Utilization

| Machine | Experiment | Est. Time |
|---------|-----------|-----------|
| 1 | 01-lr-sweep (5 LRs x 5K steps) | ~2.5hrs |
| 1 | 02-obs-dropout (3 values x 30K steps) | ~6hrs |
| 2 | 03-relative-prediction (30K steps) | ~2hrs |

Machine 2 starts 03-relative-prediction immediately with LR=3e-5 (default). If LR sweep on Machine 1 finds a better LR, update and re-run.

## Base Training Command

```bash
conda activate lerobot && export WANDB_MODE=disabled

lerobot-train \
    --policy.type=act_smooth \
    --dataset.repo_id=giacomoran/lerobot_policy_act_smooth_30fps \
    --output_dir=<output_dir> \
    --policy.input_features='{"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 640, 480], "type": "VISUAL"}, "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"}}' \
    --steps=30000 \
    --save_freq=10000 \
    --batch_size=32 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.use_vae=false \
    --policy.length_prefix_past=4 \
    --policy.length_prefix_future=2 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --num_workers=8
```
