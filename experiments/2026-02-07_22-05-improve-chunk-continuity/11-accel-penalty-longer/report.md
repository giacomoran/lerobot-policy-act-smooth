=== Results ===

```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/experiments/2026-02-07_22-05-improve-chunk-continuity/11-accel-penalty-longer/checkpoints/checkpoints/last/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30, advance=30, blend=False

=== Episode 0 (15 chunks) ===
  Predicted U_v:  2.026817  deg^2/frame^2
  Predicted U_a:  1.000113  deg^2/frame^4
  Predicted mu_v: 0.370226  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 7.76x (boundary=0.5383, non-boundary=0.0694)

=== Episode 1 (14 chunks) ===
  Predicted U_v:  1.652541  deg^2/frame^2
  Predicted U_a:  0.475021  deg^2/frame^4
  Predicted mu_v: 0.432109  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 7.12x (boundary=0.5116, non-boundary=0.0718)

=== Episode 2 (20 chunks) ===
  Predicted U_v:  1.796156  deg^2/frame^2
  Predicted U_a:  1.429100  deg^2/frame^4
  Predicted mu_v: 0.326596  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 7.11x (boundary=0.4503, non-boundary=0.0633)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.968078, std=0.390160
  pred_uniformity_velocity: mean=1.825171, std=0.154169
  pred_mean_velocity: mean=0.376311, std=0.043290
  boundary_ratio_boundary_spike: mean=7.330731, std=0.300278

```
