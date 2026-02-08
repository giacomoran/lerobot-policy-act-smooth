=== Results ===

```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/experiments/2026-02-07_22-05-improve-chunk-continuity/09-continued-lower-lr/checkpoints/checkpoints/last/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30, advance=30, blend=False

=== Episode 0 (15 chunks) ===
  Predicted U_v:  1.883136  deg^2/frame^2
  Predicted U_a:  0.126808  deg^2/frame^4
  Predicted mu_v: 0.364099  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 4.18x (boundary=0.2940, non-boundary=0.0703)

=== Episode 1 (14 chunks) ===
  Predicted U_v:  1.493623  deg^2/frame^2
  Predicted U_a:  0.146904  deg^2/frame^4
  Predicted mu_v: 0.416064  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 4.20x (boundary=0.2555, non-boundary=0.0608)

=== Episode 2 (20 chunks) ===
  Predicted U_v:  1.165127  deg^2/frame^2
  Predicted U_a:  0.127013  deg^2/frame^4
  Predicted mu_v: 0.325682  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 3.83x (boundary=0.2260, non-boundary=0.0591)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.133575, std=0.009426
  pred_uniformity_velocity: mean=1.513962, std=0.293479
  pred_mean_velocity: mean=0.368615, std=0.037036
  boundary_ratio_boundary_spike: mean=4.070662, std=0.173653

```
