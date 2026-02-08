=== Results ===

--- a20 ---
```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/experiments/2026-02-07_22-05-improve-chunk-continuity/08-continued-training/checkpoints/checkpoints/last/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30, advance=20, blend=True

=== Episode 0 (22 chunks) ===
  Predicted U_v:  1.913949  deg^2/frame^2
  Predicted U_a:  0.076536  deg^2/frame^4
  Predicted mu_v: 0.361833  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 1.33x (boundary=0.1016, non-boundary=0.0763)

=== Episode 1 (21 chunks) ===
  Predicted U_v:  1.477889  deg^2/frame^2
  Predicted U_a:  0.059733  deg^2/frame^4
  Predicted mu_v: 0.401421  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 1.28x (boundary=0.0782, non-boundary=0.0610)

=== Episode 2 (29 chunks) ===
  Predicted U_v:  1.101123  deg^2/frame^2
  Predicted U_a:  0.090183  deg^2/frame^4
  Predicted mu_v: 0.329753  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 0.96x (boundary=0.0679, non-boundary=0.0711)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.075484, std=0.012454
  pred_uniformity_velocity: mean=1.497654, std=0.332129
  pred_mean_velocity: mean=0.364336, std=0.029312
  boundary_ratio_boundary_spike: mean=1.189258, std=0.166146

```

--- a15 ---
```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/experiments/2026-02-07_22-05-improve-chunk-continuity/08-continued-training/checkpoints/checkpoints/last/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30, advance=15, blend=True

=== Episode 0 (29 chunks) ===
  Predicted U_v:  1.852625  deg^2/frame^2
  Predicted U_a:  0.064661  deg^2/frame^4
  Predicted mu_v: 0.356209  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 1.43x (boundary=0.0934, non-boundary=0.0652)

=== Episode 1 (28 chunks) ===
  Predicted U_v:  1.466537  deg^2/frame^2
  Predicted U_a:  0.066133  deg^2/frame^4
  Predicted mu_v: 0.397816  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 1.40x (boundary=0.0789, non-boundary=0.0564)

=== Episode 2 (39 chunks) ===
  Predicted U_v:  1.152980  deg^2/frame^2
  Predicted U_a:  0.094614  deg^2/frame^4
  Predicted mu_v: 0.321670  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 1.43x (boundary=0.0848, non-boundary=0.0595)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.075136, std=0.013786
  pred_uniformity_velocity: mean=1.490714, std=0.286140
  pred_mean_velocity: mean=0.358565, std=0.031131
  boundary_ratio_boundary_spike: mean=1.419952, std=0.014051

```

