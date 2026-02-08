=== Results ===

--- a30 ---
```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/050003/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30, advance=30, blend=False

=== Episode 0 (15 chunks) ===
  Predicted U_v:  1.915879  deg^2/frame^2
  Predicted U_a:  0.201946  deg^2/frame^4
  Predicted mu_v: 0.369342  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 4.62x (boundary=0.3330, non-boundary=0.0721)

=== Episode 1 (14 chunks) ===
  Predicted U_v:  1.605843  deg^2/frame^2
  Predicted U_a:  0.403183  deg^2/frame^4
  Predicted mu_v: 0.427594  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 5.50x (boundary=0.3588, non-boundary=0.0653)

=== Episode 2 (20 chunks) ===
  Predicted U_v:  1.190332  deg^2/frame^2
  Predicted U_a:  0.183271  deg^2/frame^4
  Predicted mu_v: 0.326865  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 4.88x (boundary=0.2911, non-boundary=0.0597)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.262800, std=0.099558
  pred_uniformity_velocity: mean=1.570685, std=0.297245
  pred_mean_velocity: mean=0.374600, std=0.041290
  boundary_ratio_boundary_spike: mean=4.999210, std=0.369126

```

--- a20 ---
```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/050003/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30, advance=20, blend=True

=== Episode 0 (22 chunks) ===
  Predicted U_v:  1.914485  deg^2/frame^2
  Predicted U_a:  0.060128  deg^2/frame^4
  Predicted mu_v: 0.356325  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 1.47x (boundary=0.0983, non-boundary=0.0668)

=== Episode 1 (21 chunks) ===
  Predicted U_v:  1.405178  deg^2/frame^2
  Predicted U_a:  0.046207  deg^2/frame^4
  Predicted mu_v: 0.398215  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 1.23x (boundary=0.0720, non-boundary=0.0585)

=== Episode 2 (29 chunks) ===
  Predicted U_v:  1.231017  deg^2/frame^2
  Predicted U_a:  0.080328  deg^2/frame^4
  Predicted mu_v: 0.323181  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 0.94x (boundary=0.0597, non-boundary=0.0636)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.062221, std=0.014008
  pred_uniformity_velocity: mean=1.516893, std=0.289991
  pred_mean_velocity: mean=0.359240, std=0.030702
  boundary_ratio_boundary_spike: mean=1.214535, std=0.217905

```

--- a15 ---
```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/050003/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30, advance=15, blend=True

=== Episode 0 (29 chunks) ===
  Predicted U_v:  1.890874  deg^2/frame^2
  Predicted U_a:  0.049080  deg^2/frame^4
  Predicted mu_v: 0.354839  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 1.11x (boundary=0.0730, non-boundary=0.0658)

=== Episode 1 (28 chunks) ===
  Predicted U_v:  1.404716  deg^2/frame^2
  Predicted U_a:  0.047811  deg^2/frame^4
  Predicted mu_v: 0.395094  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 1.34x (boundary=0.0703, non-boundary=0.0527)

=== Episode 2 (39 chunks) ===
  Predicted U_v:  1.202523  deg^2/frame^2
  Predicted U_a:  0.087046  deg^2/frame^4
  Predicted mu_v: 0.317808  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 1.21x (boundary=0.0689, non-boundary=0.0570)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.061312, std=0.018204
  pred_uniformity_velocity: mean=1.499371, std=0.288879
  pred_mean_velocity: mean=0.355914, std=0.031561
  boundary_ratio_boundary_spike: mean=1.218373, std=0.091869

```

--- a10 ---
```
policy: /home/giacomoran/projects/lerobot-policy-act-smooth/outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/050003/pretrained_model
dataset: giacomoran/lerobot_policy_act_smooth_30fps
episodes: [0, 1, 2]
config: K=4, D=2, C=30, n_action_steps=30, advance=10, blend=True

=== Episode 0 (43 chunks) ===
  Predicted U_v:  1.994149  deg^2/frame^2
  Predicted U_a:  0.224326  deg^2/frame^4
  Predicted mu_v: 0.368674  deg/frame
  Truth U_v:      1.800901  deg^2/frame^2
  Truth U_a:      0.173642  deg^2/frame^4
  Truth mu_v:     0.329529  deg/frame
  Boundary spike: 3.27x (boundary=0.2054, non-boundary=0.0629)

=== Episode 1 (42 chunks) ===
  Predicted U_v:  1.481722  deg^2/frame^2
  Predicted U_a:  0.241459  deg^2/frame^4
  Predicted mu_v: 0.406613  deg/frame
  Truth U_v:      1.379631  deg^2/frame^2
  Truth U_a:      0.089370  deg^2/frame^4
  Truth mu_v:     0.367974  deg/frame
  Boundary spike: 3.98x (boundary=0.2143, non-boundary=0.0539)

=== Episode 2 (58 chunks) ===
  Predicted U_v:  1.187865  deg^2/frame^2
  Predicted U_a:  0.214223  deg^2/frame^4
  Predicted mu_v: 0.332430  deg/frame
  Truth U_v:      1.060954  deg^2/frame^2
  Truth U_a:      0.093951  deg^2/frame^4
  Truth mu_v:     0.295442  deg/frame
  Boundary spike: 3.46x (boundary=0.2058, non-boundary=0.0595)

=== Aggregate ===
  pred_uniformity_acceleration: mean=0.226669, std=0.011242
  pred_uniformity_velocity: mean=1.554579, std=0.333171
  pred_mean_velocity: mean=0.369239, std=0.030288
  boundary_ratio_boundary_spike: mean=3.569429, std=0.300434

```

