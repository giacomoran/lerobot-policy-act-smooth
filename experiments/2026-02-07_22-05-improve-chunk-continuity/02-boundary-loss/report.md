=== Results ===

```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/experiments/2026-02-07_22-05-improve-chunk-continuity/02-boundary-loss/checkpoints/checkpoints/last/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30

=== Episode 0 (15 chunks) ===
  Predicted U_v:  1.924701  deg^2/frame^2
  Predicted U_a:  1.445642  deg^2/frame^4
  Predicted mu_v: 0.384460  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 8.26x (boundary=0.6075, non-boundary=0.0736)

=== Episode 1 (14 chunks) ===
  Predicted U_v:  1.583749  deg^2/frame^2
  Predicted U_a:  0.891594  deg^2/frame^4
  Predicted mu_v: 0.431671  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 6.64x (boundary=0.4925, non-boundary=0.0741)

=== Episode 2 (20 chunks) ===
  Predicted U_v:  1.330131  deg^2/frame^2
  Predicted U_a:  0.841929  deg^2/frame^4
  Predicted mu_v: 0.343210  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 7.59x (boundary=0.4958, non-boundary=0.0653)

=== Aggregate ===
  pred_uniformity_acceleration: mean=1.059722, std=0.273639
  pred_uniformity_velocity: mean=1.612860, std=0.243604
  pred_mean_velocity: mean=0.386447, std=0.036142
  boundary_ratio_boundary_spike: mean=7.497801, std=0.661585

```
