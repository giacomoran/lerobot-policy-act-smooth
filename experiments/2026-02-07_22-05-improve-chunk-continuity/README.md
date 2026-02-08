# Experiment: Improve Chunk Boundary Continuity

## Research Goal

ACTSmooth conditions action prediction on a prefix of past + committed actions to enable smooth chunk transitions. Current results on 30fps SO101 arm show ACTSmooth is 3x better than vanilla ACT at 30fps (U_a: 0.31 vs 0.92), but chunk boundary discontinuities are still visible. We want to improve further via training modifications.

## Context

| Policy            | U_a (actions) | U_a (obs) |
|-------------------|---------------|-----------|
| Vanilla ACT 10fps | 9.07          | 0.094     |
| Vanilla ACT 30fps | 0.92          | 0.117     |
| ACTSmooth 30fps   | 0.31          | 0.136     |

## Experiment Sequence

All experiments (except baseline) **fine-tune from the existing 50k-step checkpoint** for 3-5k additional steps. This is fast and tests whether each modification improves continuity on top of the already-trained model.

### 01 — Baseline
Run offline eval on current checkpoint (050003). No training. Establishes comparison metrics.

### 02 — Boundary Loss
Add C0/C1 continuity loss at prefix->chunk boundary. Fine-tune 3k steps.
- C0: `|predicted_first - last_prefix_action|`
- C1: `|velocity_after_boundary - velocity_before_boundary|`
- Try `weight_loss_boundary` in {0.01, 0.1, 1.0}

### 03 — No Proprioception
Remove `observation.state` from input_features. Train from scratch 5k steps.
Tests hypothesis that proprioception creates feedback-loop discontinuity.
(Architecture changes mean some weights don't transfer — robot_state projections are dropped.)

### 04 — Prefix Noise
Add Gaussian noise to action prefix embeddings during training. Fine-tune 3k steps.
- Try `std_noise_prefix` in {0.001, 0.01, 0.1}

### 05 — Best Combination
Combine top-performing modifications. Fine-tune 5k steps.

## How Cloned Source Works

Each experiment with code changes gets its own copy of the ACTSmooth source in `src/`. The `run.sh` script:

1. Backs up the original `src/lerobot_policy_act_smooth/` files
2. Copies the experiment's cloned versions in place
3. Runs `lerobot-train` (picks up modified code via the editable install)
4. Runs the offline eval
5. Restores the original source

This gives full freedom to make bold modifications per experiment without affecting others.

## Running Experiments

Each experiment has a `run.sh` script. Run them sequentially:

```bash
# Must be in nix develop shell with .venv activated
nix develop
source .venv/bin/activate

# Run each experiment
bash experiments/2026-02-07_22-05-improve-chunk-continuity/01-baseline/run.sh
bash experiments/2026-02-07_22-05-improve-chunk-continuity/02-boundary-loss/run.sh
# ... etc
```

After each experiment, examine the report.md and plots before proceeding to the next.

## Evaluation

All experiments are evaluated using `scripts/eval/eval_offline_replay.py` which simulates the async_smooth chunk loop on dataset episodes using ground truth actions as prefix. Key metrics:

- **U_a (uniformity_acceleration)**: Lower = smoother. Primary metric.
- **Boundary spike ratio**: Acceleration at chunk boundaries vs non-boundaries. Lower = better continuity.
- **U_v, mu_v**: Velocity metrics for additional context.

## Checkpoint

Base checkpoint: `outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/050003/`
Dataset: `giacomoran/lerobot_policy_act_smooth_30fps`
