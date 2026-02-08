=== Results ===

```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/experiments/2026-02-07_22-05-improve-chunk-continuity/07-front-weighted-loss/checkpoints/checkpoints/last/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30

=== Episode 0 (15 chunks) ===
  Predicted U_v:  1.932713  deg^2/frame^2
  Predicted U_a:  1.508282  deg^2/frame^4
  Predicted mu_v: 0.368643  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 7.28x (boundary=0.5340, non-boundary=0.0733)

=== Episode 1 (14 chunks) ===
  Predicted U_v:  1.495883  deg^2/frame^2
  Predicted U_a:  0.598476  deg^2/frame^4
  Predicted mu_v: 0.421522  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 5.32x (boundary=0.3753, non-boundary=0.0706)

=== Episode 2 (20 chunks) ===
  Predicted U_v:  1.476121  deg^2/frame^2
  Predicted U_a:  0.911481  deg^2/frame^4
  Predicted mu_v: 0.331783  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 7.10x (boundary=0.4549, non-boundary=0.0640)

=== Aggregate ===
  pred_uniformity_acceleration: mean=1.006080, std=0.377402
  pred_uniformity_velocity: mean=1.634906, std=0.210736
  pred_mean_velocity: mean=0.373983, std=0.036830
  boundary_ratio_boundary_spike: mean=6.567940, std=0.885861

```
