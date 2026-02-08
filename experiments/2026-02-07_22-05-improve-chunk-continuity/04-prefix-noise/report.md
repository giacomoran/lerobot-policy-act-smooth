=== Results ===

```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/experiments/2026-02-07_22-05-improve-chunk-continuity/04-prefix-noise/checkpoints/checkpoints/last/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30

=== Episode 0 (15 chunks) ===
  Predicted U_v:  1.957600  deg^2/frame^2
  Predicted U_a:  0.286300  deg^2/frame^4
  Predicted mu_v: 0.370802  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 4.69x (boundary=0.3732, non-boundary=0.0795)

=== Episode 1 (14 chunks) ===
  Predicted U_v:  1.882053  deg^2/frame^2
  Predicted U_a:  1.846654  deg^2/frame^4
  Predicted mu_v: 0.427972  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 7.08x (boundary=0.5145, non-boundary=0.0727)

=== Episode 2 (20 chunks) ===
  Predicted U_v:  1.195497  deg^2/frame^2
  Predicted U_a:  0.365245  deg^2/frame^4
  Predicted mu_v: 0.329689  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 6.42x (boundary=0.4183, non-boundary=0.0651)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.832733, std=0.717674
  pred_uniformity_velocity: mean=1.678383, std=0.342843
  pred_mean_velocity: mean=0.376154, std=0.040302
  boundary_ratio_boundary_spike: mean=6.066609, std=1.007747

```
