=== Results ===

```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/experiments/2026-02-07_22-05-improve-chunk-continuity/08-continued-training/checkpoints/checkpoints/last/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30

=== Episode 0 (15 chunks) ===
  Predicted U_v:  1.974180  deg^2/frame^2
  Predicted U_a:  1.099131  deg^2/frame^4
  Predicted mu_v: 0.368948  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 5.20x (boundary=0.3975, non-boundary=0.0764)

=== Episode 1 (14 chunks) ===
  Predicted U_v:  1.575321  deg^2/frame^2
  Predicted U_a:  0.337491  deg^2/frame^4
  Predicted mu_v: 0.426369  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 5.48x (boundary=0.3897, non-boundary=0.0711)

=== Episode 2 (20 chunks) ===
  Predicted U_v:  1.423188  deg^2/frame^2
  Predicted U_a:  0.761012  deg^2/frame^4
  Predicted mu_v: 0.332811  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 6.04x (boundary=0.4201, non-boundary=0.0696)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.732545, std=0.311589
  pred_uniformity_velocity: mean=1.657563, std=0.232337
  pred_mean_velocity: mean=0.376043, std=0.038523
  boundary_ratio_boundary_spike: mean=5.572322, std=0.348413

```
