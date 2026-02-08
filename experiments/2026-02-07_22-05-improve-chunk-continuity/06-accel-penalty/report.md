=== Results ===

```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/experiments/2026-02-07_22-05-improve-chunk-continuity/06-accel-penalty/checkpoints/checkpoints/last/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30

=== Episode 0 (15 chunks) ===
  Predicted U_v:  1.981209  deg^2/frame^2
  Predicted U_a:  0.304331  deg^2/frame^4
  Predicted mu_v: 0.375336  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 4.85x (boundary=0.3801, non-boundary=0.0784)

=== Episode 1 (14 chunks) ===
  Predicted U_v:  1.657951  deg^2/frame^2
  Predicted U_a:  0.755183  deg^2/frame^4
  Predicted mu_v: 0.440355  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 7.08x (boundary=0.5194, non-boundary=0.0734)

=== Episode 2 (20 chunks) ===
  Predicted U_v:  1.232714  deg^2/frame^2
  Predicted U_a:  0.501970  deg^2/frame^4
  Predicted mu_v: 0.332286  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 6.46x (boundary=0.4209, non-boundary=0.0652)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.520495, std=0.184525
  pred_uniformity_velocity: mean=1.623958, std=0.306516
  pred_mean_velocity: mean=0.382659, std=0.044422
  boundary_ratio_boundary_spike: mean=6.127420, std=0.938215

```
