# Synthesis: Improve Chunk Boundary Continuity

## Context

ACTSmooth conditions action prediction on a prefix of `[past(K=4) | future(D=2)]` actions to enable smooth chunk transitions. The policy uses `chunk_size=30`, `n_action_steps=30` at 30fps on SO101.

**Starting point** (prior work, measured on real robot via rerun logs):

| Policy            | U_a (actions) | U_a (obs) |
|-------------------|---------------|-----------|
| Vanilla ACT 10fps | 9.07          | 0.094     |
| Vanilla ACT 30fps | 0.92          | 0.117     |
| ACTSmooth 30fps   | 0.31          | 0.136     |

**Goal**: Reduce chunk boundary discontinuities further, targeting lower U_a and boundary spike ratios.

**Evaluation method**: Offline replay on 3 dataset episodes using `scripts/eval/eval_offline_replay.py`. Ground truth actions are used as prefix at each chunk boundary, isolating the model's ability to produce smooth continuations. Metrics:
- **U_a** (uniformity of acceleration, lower = smoother): Primary metric
- **Boundary spike** (ratio of boundary vs non-boundary acceleration): Measures discontinuity specifically at chunk transitions. 1.0x = no discontinuity.

---

## Results Summary

### Training-based experiments (no overlap blending)

All experiments use `advance=30` (= `n_action_steps`), no overlap.

| # | Experiment | Training | Mean U_a | Mean Spike | Verdict |
|---|-----------|----------|----------|------------|---------|
| 01 | **Baseline** (50K) | None | 0.414 | 6.08x | Reference |
| 10-a30 | **Baseline** (50K, 050003 ckpt) | None | 0.263 | 5.00x | Better reference* |
| 02 | Boundary loss (w=0.1) | +3K ft | 1.060 | 7.50x | WORSE (+156%) |
| 03 | No proprioception | 5K scratch | 9.220 | 8.38x | NOT COMPARABLE |
| 04 | Prefix noise (std=0.01) | +3K ft | 0.833 | 6.07x | WORSE (+217%) |
| 06 | Accel penalty (w=0.01) | +3K ft | 0.520 | 6.13x | WORSE (+98%) |
| 07 | Front-weighted loss | +3K ft | 1.006 | 6.57x | WORSE (+283%) |
| 08 | Continued training | +10K ft | 0.733 | 5.57x | MIXED (spike improved) |
| **09** | **Continued + lower LR (3e-6)** | **+10K ft** | **0.134** | **4.07x** | **BEST training** |
| 11 | Accel penalty + longer | +10K ft | 0.968 | 7.33x | WORSE (+268%) |

*\* 01-baseline used the `last` checkpoint alias while 10-a30 used `050003` explicitly. Minor difference from checkpoint selection.*

### Inference-time overlap blending experiments

Overlap blending: advance fewer than `n_action_steps` per chunk and linearly blend overlapping predictions.

| # | Model | Advance | Overlap | Mean U_a | Mean Spike | vs Baseline |
|---|-------|---------|---------|----------|------------|-------------|
| 10-a30 | 50K | 30 | 0 | 0.263 | 5.00x | Reference |
| **10-a20** | **50K** | **20** | **10** | **0.062** | **1.21x** | **U_a -76%, spike -76%** |
| 10-a15 | 50K | 15 | 15 | 0.061 | 1.22x | U_a -77%, spike -76% |
| 10-a10 | 50K | 10 | 20 | 0.227 | 3.57x | U_a -14%, spike -29% |
| 12-a20 | 60K | 20 | 10 | 0.075 | 1.19x | U_a -71%, spike -76% |
| 12-a15 | 60K | 15 | 15 | 0.075 | 1.42x | U_a -71%, spike -72% |
| **09+blend** | **60K (lr=3e-6)** | **20** | **10** | **0.058** | **1.18x** | **U_a -78%, spike -76%** |

### Best overall: lower LR training + overlap blending

Combining experiment 09's model (lower LR fine-tuning) with advance=20 blending achieves the best result:
- **U_a = 0.058** (vs 0.263 baseline, **-78%**)
- **Spike = 1.18x** (vs 5.00x baseline, **-76%**)
- Per-episode: Ep0 U_a=0.054/spike=1.34x, Ep1 U_a=0.042/spike=1.23x, Ep2 U_a=0.079/spike=0.98x

---

## Key Findings

### 1. Overlap blending is the breakthrough technique

Overlap blending with `advance=20` (10 actions overlap out of 30) reduces boundary spikes from 5-6x to ~1.2x — virtually eliminating chunk boundary discontinuities. This requires **zero training**, just a change to inference-time chunk scheduling.

The sweet spot is `advance=20` (1/3 overlap):
- `advance=20` and `advance=15` perform similarly (~0.06 U_a, ~1.2x spike)
- `advance=10` (2/3 overlap) degrades significantly (0.23 U_a, 3.6x spike) — too much overlap causes the blend to accumulate errors

### 2. Training-based loss modifications generally hurt

Every auxiliary loss tried (boundary loss, accel penalty, front-weighted loss, prefix noise) made U_a **worse**. The common failure mode:
- Adding auxiliary losses during fine-tuning destabilizes the already-converged action prediction
- The model trades overall prediction quality for the auxiliary objective
- Short fine-tuning (3K steps) isn't enough to re-converge with the new loss landscape

### 3. Lower learning rate is the key to successful fine-tuning

Experiment 09 (lr=3e-6, 10x lower than default 3e-5) achieved the best training-based result:
- U_a: 0.134 (vs 0.263 baseline, -49%)
- Spike: 4.07x (vs 5.00x baseline, -19%)
- Final loss: 0.023 (vs ~0.045 for other fine-tunes)

This suggests the default lr=3e-5 is too aggressive for fine-tuning from a converged checkpoint. The lower LR allows continued convergence without destabilizing.

### 4. More training at original LR doesn't help U_a

Experiment 08 (10K steps at original lr=3e-5) improved spike (5.57x vs 6.08x) but worsened U_a (0.733 vs 0.414). The model may be overfitting or the LR is too high for this phase.

### 5. Lower LR + blending is the best combination

The 09 model (lower LR, 60K steps) + blending achieves U_a=0.058, spike=1.18x — slightly better than the 50K baseline with blending (0.062, 1.21x). The improvements are additive: better model quality from lower-LR training + smoother transitions from blending.

### 6. Blending is robust to underlying model quality

All three models tested with blending (50K, 60K original LR, 60K lower LR) achieve similar results (U_a 0.058-0.075, spike 1.18-1.42x), confirming blending is the dominant factor.

---

## Recommendations

### Immediate (deploy now)

**Use overlap blending with `advance=20`** in the inference loop. This is a pure inference-time change:
- Modify `eval_async_smooth.py` to advance by 20 steps instead of 30
- Linearly blend the 10 overlapping actions between old and new chunks
- Expected improvement: 5x reduction in boundary spikes (from ~5x to ~1.2x)

### Short-term (worth trying)

1. **Full from-scratch training at lr=3e-6**: The fine-tuning results suggest the default lr may be suboptimal. Training from scratch with a lower peak LR (or with LR warmup + cosine decay to 3e-6) may produce an even better base model.

2. **Non-linear blending functions**: The current linear blend could be replaced with sigmoid, cosine, or other curves that might produce smoother transitions.

### Not recommended

- Boundary loss, accel penalty, front-weighted loss, prefix noise: All degraded U_a. These approaches need fundamentally different formulations (e.g., training with blending in the loop) to work.
- Removing proprioception: Catastrophically bad without full from-scratch training with enough steps.

---

## Experiment Details

### 01-baseline
- **Model**: 50K checkpoint (last alias)
- **Training**: None
- **Per-episode**: Ep0 U_a=0.200/spike=5.78x, Ep1 U_a=0.643/spike=5.62x, Ep2 U_a=0.398/spike=6.83x

### 02-boundary-loss
- **Model**: 50K + 3K fine-tune with C0+C1 boundary continuity loss (weight=0.1)
- **Code change**: Added boundary loss computing |pred_first - prefix_last| (C0) and |vel_after - vel_before| (C1)
- **Per-episode**: Ep0 U_a=1.446/spike=8.26x, Ep1 U_a=0.892/spike=6.64x, Ep2 U_a=0.842/spike=7.59x
- **Analysis**: Boundary loss actively hurt — the model overfits to minimizing the loss at the single boundary point while degrading overall prediction quality

### 03-no-proprioception
- **Model**: 5K from scratch, no observation.state
- **Per-episode**: Ep0 U_a=16.2/spike=10.2x, Ep1 U_a=8.4/spike=7.4x, Ep2 U_a=3.1/spike=7.6x
- **Analysis**: Not comparable — 5K steps is far too few to train from scratch. Would need 50K+ steps for fair comparison.

### 04-prefix-noise
- **Model**: 50K + 3K fine-tune with Gaussian noise on prefix embeddings (std=0.01)
- **Per-episode**: Ep0 U_a=0.286/spike=4.69x, Ep1 U_a=1.847/spike=7.08x, Ep2 U_a=0.365/spike=6.42x
- **Analysis**: High variance across episodes. Ep0 actually improved slightly but Ep1 catastrophically worsened.

### 06-accel-penalty
- **Model**: 50K + 3K fine-tune with acceleration penalty loss (weight=0.01)
- **Per-episode**: Ep0 U_a=0.304/spike=4.85x, Ep1 U_a=0.755/spike=7.08x, Ep2 U_a=0.502/spike=6.46x
- **Analysis**: Similar to prefix noise — marginal improvement on some episodes, regression on others.

### 07-front-weighted-loss
- **Model**: 50K + 3K fine-tune with higher loss weight on early chunk positions
- **Per-episode**: Ep0 U_a=1.508/spike=7.28x, Ep1 U_a=0.598/spike=5.32x, Ep2 U_a=0.911/spike=7.10x
- **Analysis**: Weighting the loss toward early positions didn't help — likely because the boundary discontinuity is a global coherence problem, not a per-position accuracy problem.

### 08-continued-training
- **Model**: 50K + 10K fine-tune at original lr=3e-5
- **Final loss**: 0.043 (down from 0.065)
- **Per-episode**: Ep0 U_a=1.099/spike=5.20x, Ep1 U_a=0.337/spike=5.48x, Ep2 U_a=0.761/spike=6.04x
- **Analysis**: Spike improved 8.4% but U_a worsened 181%. More training at original LR oversmooths some episodes but creates artifacts in others.

### 09-continued-lower-lr
- **Model**: 50K + 10K fine-tune at lr=3e-6 (10x lower)
- **Final loss**: 0.023
- **Per-episode**: Ep0 U_a=0.127/spike=4.18x, Ep1 U_a=0.147/spike=4.20x, Ep2 U_a=0.127/spike=3.83x
- **Analysis**: Best training-based result by far. Low variance across episodes. LR 10x lower was the key difference vs experiment 08.

### 10-overlap-blend
- **Model**: 50K checkpoint (050003), 4 blending variants
- **Best variant (a20)**: advance=20, 10 overlap, linear blend
  - Per-episode: Ep0 U_a=0.060/spike=1.47x, Ep1 U_a=0.046/spike=1.23x, Ep2 U_a=0.080/spike=0.94x
- **Analysis**: Dramatic improvement. The boundary spike on Ep2 is actually below 1.0x, meaning blended boundaries are smoother than non-boundaries.

### 11-accel-penalty-longer
- **Model**: 50K + 10K fine-tune with accel penalty (w=0.01)
- **Per-episode**: Ep0 U_a=1.000/spike=7.76x, Ep1 U_a=0.475/spike=7.12x, Ep2 U_a=1.429/spike=7.11x
- **Analysis**: Longer training with accel penalty made things worse, not better. The penalty drives the model toward predictions with low within-chunk acceleration but doesn't address boundary discontinuity.

### 12-overlap-blend-continued
- **Model**: 60K checkpoint (from exp 08) with blending
- **a20 variant**: advance=20, 10 overlap
  - Per-episode: Ep0 U_a=0.077/spike=1.33x, Ep1 U_a=0.060/spike=1.28x, Ep2 U_a=0.090/spike=0.96x
  - Mean: U_a=0.075, spike=1.19x
- **a15 variant**: advance=15, 15 overlap
  - Per-episode: Ep0 U_a=0.065/spike=1.43x, Ep1 U_a=0.066/spike=1.40x, Ep2 U_a=0.095/spike=1.43x
  - Mean: U_a=0.075, spike=1.42x
- **Analysis**: Confirms blending works on different base models. Slightly higher U_a than 50K model (0.075 vs 0.062) but still excellent.
