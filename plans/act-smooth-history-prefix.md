# ACTSmooth with History Prefix

## Core Problems

### 1. Chunk Continuity

Action chunking policies predict sequences of actions that are executed over time. When switching from one chunk to the next, there can be **discontinuities** (jerks) in the robot motion if the new chunk doesn't smoothly continue from where the previous chunk left off.

### 2. Inference Latency

Policy inference is not instantaneous (~33ms on our hardware). During inference, the robot must continue executing actions from the previous chunk. The new chunk must account for this: it should predict actions that continue from where the robot will be when inference completes, not where it was when inference started.

### 3. Context vs Reactivity Tradeoff

To ensure continuity, the policy needs context about recent/committed actions. But in current approaches (Training-Time RTC), more context = higher delay = slower reaction to changes. We need to **decouple context from delay**.

## Current Approach Limitations

Current ACTSmooth (and Training-Time RTC) conflates two concepts:

- **Context length**: How much action history for continuity
- **Delay**: How many committed future actions (determines when new chunk starts)

With the current design, `prefix = committed future actions only`, so:

- More context → higher delay → worse reactivity

This creates a problem at higher frame rates:

- At 10fps with delay=2: 200ms of context, 200ms delay
- At 30fps with delay=2: 67ms of context, 67ms delay

The 30fps policy has much less context, which may explain why it "ignores" the prefix - there's simply not enough temporal information.

## How We Arrived at This Solution

### Initial observations

1. 10fps policy with interpolation to 30fps works well - smooth and reactive
2. 30fps policy with RTC (delay=2) is not smooth, seems to ignore prefix
3. Delay sensitivity plots show 10fps responds to prefix, 30fps less so

### Ideas considered and rejected

1. **State rollforward (VLASH-style)**: Compute future robot state by integrating prefix actions. Rejected because it's synthetic - doesn't account for execution noise, motor dynamics, etc.

2. **Increase prefix_length_future_max for 30fps**: Training with prefix_length_future_max=6 at 30fps would give 200ms context. But with ~33ms inference latency, actual delay used at inference would still be ~1-2. Training on delays you won't use doesn't help.

3. **Drop observation state**: Simpler but doesn't address the core context problem.

### The insight

Decouple context from delay by including **past executed actions** in the prefix:

```
Prefix = [past completed actions] + [committed pending actions]
         ├─── k actions (fixed) ──┤  ├─── d actions (variable, d ≥ 1) ─┤
```

Example at 30fps with k=4, d=2:

- Prefix = [4 past actions, 2 committed] = 6 total (200ms context)
- But delay = only 2 (67ms until new chunk starts)
- **Best of both**: rich context for continuity + fast reactivity

## Semantics

### Notation

- `t`: observation timestep (t=0 is the current observation)
- `o_t`: observation at timestep t
- `a_t`: action corresponding to timestep t
- `k` (`prefix_length_past`): number of past (completed) actions in prefix (fixed hyperparameter)
- `d`: number of committed (pending) actions in prefix (variable, 1 ≤ d ≤ `prefix_length_future_max`)

### Key insight: t_0 is always committed

During data collection at t_0:

- Observation is the current robot position
- Action is the command being sent right now

During inference at t_0:

- Observation is the current robot position
- But the output action won't execute until the next frame (non-zero inference latency)
- Therefore, the action at t_0 is ALWAYS from the previous chunk - it's already committed

**Implication**: `a_{t_0}` must always be in the prefix, never in the target. We enforce `d ≥ 1`.

### Training data layout

For observation at timestep `t_0`:

```
Past actions:      [a_{t_{-k}}, ..., a_{t_{-1}}]           (k actions, completed)
Committed actions: [a_{t_0}, ..., a_{t_{d-1}}]             (d actions, d ≥ 1, pending)
Target:            [a_{t_d}, ..., a_{t_{d+chunk_size-1}}]  (chunk_size actions)
```

Total prefix length = `k + d` (variable due to d, but always ≥ k + 1)

### Why this split makes sense

- **Past (completed)**: Actions `[t_{-k}, ..., t_{-1}]` have finished executing. We know exactly what the robot did.
- **Committed (pending)**: Actions `[t_0, ..., t_{d-1}]` are committed but haven't completed:
  - `a_{t_0}` is being sent/executed RIGHT NOW
  - `a_{t_1}, ...` will execute during inference
- **Target**: Actions `[t_d, ...]` are what we need to predict - the continuation after committed actions complete.

### Variable d, fixed k

During training:

- `k` is fixed (e.g., 4 for ~133ms at 30fps, 1 for ~100ms at 10fps)
- `d` is sampled from `{1, 2, ..., prefix_length_future_max}` for each batch (note: d ≥ 1, not d ≥ 0)
- This provides consistent history context while handling variable inference timing

At inference:

- Maintain a buffer of the last `k` executed actions (completed)
- Get remaining `d` actions from current chunk (committed/pending)
- `d` naturally varies based on when inference is triggered and actual latency

## Example Configurations

Aiming for ~200ms total prefix context:

### 10fps (k=1, d∈{1})

```
obs: t_0
past actions:      [t_{-1}]                    (100ms completed)
committed actions: [t_0]                       (100ms pending)
target:            [t_1, ...]                  (chunk_size actions)

Total prefix: 200ms context, 100ms delay
```

### 30fps (k=4, d∈{1,2})

```
obs: t_0
past actions:      [t_{-4}, t_{-3}, t_{-2}, t_{-1}]    (133ms completed)
committed actions: [t_0] or [t_0, t_1]                  (33-67ms pending)
target:            [t_1, ...] or [t_2, ...]             (chunk_size actions)

Total prefix: 167-200ms context, 33-67ms delay
```

## Implementation Plan

### 1. Configuration changes (`configuration_act_smooth.py`)

Add new config parameter:

```python
# Number of past (completed) actions to include in prefix (fixed)
# These provide historical context for continuity without adding delay
prefix_length_past: int = 0  # 0 = current behavior (no history)
```

and rename `max_delay` to `prefix_length_future_max`.

Modify `action_delta_indices`:

```python
@property
def action_delta_indices(self) -> list:
    # Load: past history + committed actions + targets
    #
    # Data layout in batch[ACTION]:
    #   [t_{-k}, ..., t_{-1}, t_0, t_1, ..., t_{prefix_length_future_max-1}, t_{prefix_length_future_max}, ..., t_{prefix_length_future_max+chunk_size-1}]
    #   |---- k past -----|  |---- prefix_length_future_max committed ----|  |-------- chunk_size target --------|
    #
    # Indices relative to observation (t_0 = index 0):
    #   [-k, ..., -1, 0, 1, ..., prefix_length_future_max + chunk_size - 1]
    #
    # Note: t_0 is always committed (d >= 1), so target starts at t_d where d >= 1
    start = -self.prefix_length_past
    end = self.prefix_length_future_max + self.chunk_size
    return list(range(start, end))
```

Update validation to enforce d >= 1:

```python
def __post_init__(self):
    ...
    if self.prefix_length_future_max < 1:
        raise ValueError(
            f"prefix_length_future_max must be >= 1 (t_0 is always committed due to inference latency). "
            f"Got {self.prefix_length_future_max}."
        )
```

### 2. Model changes (`modeling_act_smooth.py`)

#### 2.1 Update `_indices_target` buffer

Current: `_indices_target[d, t] = d + t` maps delay d and chunk position t to action index
New: Account for history offset, and d starts at 1 (not 0)

```python
# In __init__:
n_history = config.prefix_length_past

# _indices_target[d, t] = n_history + d + t
# where d is in {1, ..., prefix_length_future_max} (not {0, ..., prefix_length_future_max})
# This maps (delay, chunk_position) -> index in batch[ACTION]
#
# For d=1: targets are [t_1, t_2, ...] -> indices [n_history+1, n_history+2, ...]
# For d=2: targets are [t_2, t_3, ...] -> indices [n_history+2, n_history+3, ...]
indices_delay = torch.arange(1, config.prefix_length_future_max + 1, dtype=torch.long)  # [1, 2, ..., prefix_length_future_max]
indices_offset = torch.arange(config.chunk_size, dtype=torch.long)
self.register_buffer(
    "_indices_target",
    n_history + indices_delay.unsqueeze(1) + indices_offset.unsqueeze(0)
)

# Indices for history (past completed actions)
# Maps to batch[ACTION][:, 0:n_history] which contains [t_{-k}, ..., t_{-1}]
self.register_buffer(
    "_indices_history",
    torch.arange(n_history, dtype=torch.long)
)

# Indices for committed actions need to account for variable d
# For d in {1, ..., prefix_length_future_max}, committed = [t_0, ..., t_{d-1}]
# In batch[ACTION], t_0 is at index n_history
```

#### 2.2 Update encoder input construction

Token ordering in encoder: `[images, latent, env_state, robot_state, history, committed_prefix]`

History tokens come BEFORE committed prefix tokens (temporal order: past → future).
Both use the same `encoder_action_input_proj` (shared projection).

```python
n_history = self.config.prefix_length_past
if n_history > 0:
    # Get history actions and apply mask (see section 2.6 for masking details)
    # ...history_embed with torch.where for pad positions...

    history_tokens = history_embed.permute(1, 0, 2)  # [n_history, B_eff, dim]
    encoder_in_tokens = torch.cat([encoder_in_tokens, history_tokens], dim=0)

    # Positional embeddings for history (continuous with committed prefix)
    history_pos_embed = self.encoder_1d_feature_pos_embed.weight[
        pos_1d_idx : pos_1d_idx + n_history
    ]
    history_pos_embed = history_pos_embed.unsqueeze(1).expand(-1, batch_size_effective, -1)
    encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, history_pos_embed], dim=0)
    pos_1d_idx += n_history

# Then add committed prefix tokens (after history)
# ... existing prefix code ...
```

#### 2.3 Update position embedding allocation

```python
# In __init__, update n_1d_tokens:
n_1d_tokens = 1  # latent
if self.config.robot_state_feature:
    n_1d_tokens += 1
if self.config.env_state_feature:
    n_1d_tokens += 1
n_1d_tokens += config.prefix_length_past            # action history (fixed, past)
n_1d_tokens += config.prefix_length_future_max      # action prefix (variable, committed)

self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

# No separate projection needed - history uses encoder_action_input_proj (shared)
```

#### 2.4 Update forward loss computation

```python
# In forward(), update target extraction:
# batch[ACTION] now has shape [B, n_history + prefix_length_future_max + chunk_size, action_dim]
# Targets start at index n_history + d for delay d

n_history = self.config.prefix_length_past
targets = batch[ACTION][:, self._indices_target]  # Already accounts for n_history offset
```

#### 2.5 Update predict_action_chunk for inference

```python
def predict_action_chunk(
    self,
    batch: dict[str, Tensor],
    action_prefix: Tensor | None = None,
    action_history: Tensor | None = None,  # [B, n_valid, action_dim], n_valid <= prefix_length_past
) -> Tensor:
    """
    Args:
        action_prefix: [B, delay, action_dim] - committed future actions
        action_history: [B, n_valid, action_dim] - past executed actions
                        Can have n_valid < prefix_length_past at episode start
                        If None, all history positions use action_prefix_pad_embed
    """
    n_history = self.config.prefix_length_past

    # Communicate missing history positions via batch["history_is_pad"]
    if n_history > 0:
        if action_history is None:
            n_valid = 0
        else:
            n_valid = action_history.shape[1]

        # Pad to full length with zeros (will be masked out by history_is_pad)
        if n_valid < n_history:
            pad = torch.zeros(
                batch_size, n_history - n_valid, action_dim,
                device=device, dtype=dtype
            )
            if action_history is None:
                action_history_padded = pad
            else:
                # Pad goes at the START (older positions are missing first)
                action_history_padded = torch.cat([pad, action_history], dim=1)
        else:
            action_history_padded = action_history

        # Build mask: first (n_history - n_valid) positions are invalid
        history_is_pad = torch.zeros(batch_size, n_history, dtype=torch.bool, device=device)
        history_is_pad[:, : n_history - n_valid] = True

        # Store in batch for forward pass
        batch["action_history"] = action_history_padded
        batch["history_is_pad"] = history_is_pad
```

#### 2.6 Update forward to use history_is_pad mask

```python
# In forward(), after projecting history actions:
if n_history > 0:
    history_actions = batch.get("action_history", batch[ACTION][:, :n_history])
    history_embed = self.encoder_action_input_proj(history_actions)

    # Replace padded positions with learned embedding
    if "history_is_pad" in batch:
        history_is_pad = batch["history_is_pad"]
        # Expand for training if needed (same pattern as prefix)
        if is_training_batch and self.training:
            history_is_pad = (
                history_is_pad.unsqueeze(1)
                .expand(-1, prefix_length_future_max, -1)
                .reshape(batch_size_effective, n_history)
            )
        history_embed = torch.where(
            history_is_pad.unsqueeze(-1),
            self.action_prefix_pad_embed[None, None, :].expand_as(history_embed),
            history_embed,
        )

    history_tokens = history_embed.permute(1, 0, 2)  # [n_history, B_eff, dim]
    encoder_in_tokens = torch.cat([encoder_in_tokens, history_tokens], dim=0)
```

### 3. Eval script changes (`eval_async_rtc.py`)

#### 3.1 Maintain action history buffer

```python
# In thread_actor_fn or State:
action_history_buffer: deque[Tensor] = deque(maxlen=prefix_length_past)

# After executing each action:
action_history_buffer.append(action_tensor)
```

#### 3.2 Pass history to inference

```python
# In thread_inference_fn:
with state.lock:
    action_history = torch.stack(list(state.action_history_buffer), dim=0)
    action_history = action_history.unsqueeze(0)  # [1, k, action_dim]

actions = policy.predict_action_chunk(
    observation,
    action_prefix=action_prefix_absolute,
    action_history=action_history,
)
```

#### 3.3 Handle warmup / initial episodes

At episode start, history buffer is empty or partial:

- Pass `action_history=None` or partial tensor (n_valid < k) to `predict_action_chunk`
- Model uses `action_prefix_pad_embed` directly for missing history positions
- Valid history positions get projected through `encoder_action_input_proj`
- Model should be robust to this from training (early timesteps in episodes have same pattern)

### 4. Training script changes

Update training configs:

```bash
--policy.prefix_length_past=4  # For 30fps: 4 * 33ms = 133ms past context
--policy.prefix_length_future_max=2         # d ∈ {1, 2}, covering 33-67ms committed
```

Suggested configurations (aiming for ~200ms total prefix):

| FPS | k (prefix_length_past) | prefix_length_future_max | d range | Past context | Committed | Total prefix |
| --- | ---------------------- | ------------------------ | ------- | ------------ | --------- | ------------ |
| 10  | 1                      | 2                        | {1, 2}  | 100ms        | 100-200ms | 200-300ms    |
| 30  | 4                      | 2                        | {1, 2}  | 133ms        | 33-67ms   | 167-200ms    |

Note: `d` is sampled from `{1, ..., prefix_length_future_max}` during training (not `{0, ..., prefix_length_future_max}`).

## Testing Plan

### Unit tests

1. Verify action_delta_indices returns correct range
2. Verify \_indices_target correctly maps to targets
3. Verify forward pass with prefix_length_past > 0

### Integration tests

1. Train on small dataset with prefix_length_past=2
2. Verify loss decreases
3. Run inference and verify shapes

### Ablation experiments

1. 30fps, prefix_length_past=0, prefix_length_future_max=2 (current ACTSmooth baseline)
2. 30fps, prefix_length_past=4, prefix_length_future_max=2 (proposed: ~200ms total context)
3. 10fps, prefix_length_past=1, prefix_length_future_max=2 (proposed: ~200-300ms total context)
4. Compare delay sensitivity plots and chunk transition smoothness

## Backward Compatibility

- `prefix_length_past=0` reproduces current ACTSmooth behavior exactly
- Existing checkpoints remain valid (they have prefix_length_past=0 implicitly)
- Can't load new checkpoints (prefix_length_past>0) into old code

## Design Decisions

1. **History uses same projection as prefix (shared)**
   - Simpler, fewer parameters
   - Both are action sequences, semantic difference handled by positional encoding

2. **Shared sequence positional encoding for [history, committed]**
   - Continuous positions across the full prefix
   - Temporal ordering is meaningful (past → present → future)

3. **History padding uses existing `action_prefix_pad_embed`**
   - At episode start when history buffer is partial, use the same padding as prefix
   - `predict_action_chunk` handles this when history is unavailable

4. **VAE encoder does NOT see history**
   - Keep VAE encoder similar to vanilla ACT
   - User typically disables VAE during training anyway

5. **Token ordering: [images, latent, env_state, robot_state, history, committed_prefix]**
   - History before committed makes temporal sense (past → present → future)
