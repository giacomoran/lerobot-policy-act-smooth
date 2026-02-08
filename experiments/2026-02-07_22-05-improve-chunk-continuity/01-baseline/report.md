=== Results ===

```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/last/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30

=== Episode 0 (15 chunks) ===
  Predicted U_v:  1.897259  deg^2/frame^2
  Predicted U_a:  0.200128  deg^2/frame^4
  Predicted mu_v: 0.376619  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 5.78x (boundary=0.4373, non-boundary=0.0757)

=== Episode 1 (14 chunks) ===
  Predicted U_v:  1.614525  deg^2/frame^2
  Predicted U_a:  0.643231  deg^2/frame^4
  Predicted mu_v: 0.442857  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 5.62x (boundary=0.4802, non-boundary=0.0854)

=== Episode 2 (20 chunks) ===
  Predicted U_v:  1.222679  deg^2/frame^2
  Predicted U_a:  0.397582  deg^2/frame^4
  Predicted mu_v: 0.335934  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 6.83x (boundary=0.4786, non-boundary=0.0700)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.413647, std=0.181252
  pred_uniformity_velocity: mean=1.578154, std=0.276594
  pred_mean_velocity: mean=0.385137, std=0.044065
  boundary_ratio_boundary_spike: mean=6.077208, std=0.539358

```
