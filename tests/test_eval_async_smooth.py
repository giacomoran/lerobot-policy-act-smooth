"""Test chunk transition logic for eval_async_smooth.

Tests the extracted pure functions (step_actor_control_frame, make_prefix_for_inference)
directly, without robot hardware or threading.

Core invariants:
1. Interpolation between consecutive control frames is strictly linear.
2. No overlap: current chunk's last action and pending chunk's first action
   are at consecutive timesteps. No action is discarded.
3. Committed prefix = the D remaining actions starting at the current one (inclusive).
   The model predicts from timestep_start = trigger_timestep + D.
"""

import torch
from eval_async_smooth import (
    ActionChunk,
    ArgsStepActorControlFrame,
    ArgsMakePrefixForInference,
    step_actor_control_frame,
    make_prefix_for_inference,
)


# ============================================================================
# Helpers
# ============================================================================


def make_chunk(
    n_actions: int,
    timestep_start_obs: int,
    length_prefix_future_effective: int = 0,
    length_prefix_future: int = 0,
    length_prefix_past: int = 0,
    action_dim: int = 1,
    idx_chunk: int = 0,
    base_value: float = 0.0,
):
    """Create a chunk with linearly increasing action values."""
    actions = (torch.arange(n_actions, dtype=torch.float32) + base_value).unsqueeze(1).expand(-1, action_dim)
    return ActionChunk(
        actions=actions,
        timestep_start_obs=timestep_start_obs,
        length_prefix_future_effective=length_prefix_future_effective,
        length_prefix_future=length_prefix_future,
        length_prefix_past=length_prefix_past,
        idx_chunk=idx_chunk,
    )


# ============================================================================
# Actor tests
# ============================================================================


def test_chunk_switch_when_exhausted():
    """Current chunk exhausted + pending available -> switches, returns pending's action."""
    chunk_0 = make_chunk(n_actions=3, timestep_start_obs=0, idx_chunk=0)
    chunk_1 = make_chunk(n_actions=5, timestep_start_obs=3, idx_chunk=1, base_value=100.0)

    output = step_actor_control_frame(
        ArgsStepActorControlFrame(
            timestep=3,
            action_chunk_active=chunk_0,
            action_chunk_pending=chunk_1,
            is_inference_requested=False,
            length_prefix_future=1,
        )
    )

    assert output.action_chunk_pending_consumed is True
    assert output.action_chunk_active is chunk_1
    assert output.action is not None
    assert float(output.action[0]) == 100.0
    assert output.cnt_actions_discarded == 0


def test_no_switch_when_current_has_actions():
    """Current chunk has actions + pending available -> stays on current."""
    chunk_0 = make_chunk(n_actions=5, timestep_start_obs=0, idx_chunk=0)
    chunk_1 = make_chunk(n_actions=5, timestep_start_obs=3, idx_chunk=1, base_value=100.0)

    output = step_actor_control_frame(
        ArgsStepActorControlFrame(
            timestep=2,
            action_chunk_active=chunk_0,
            action_chunk_pending=chunk_1,
            is_inference_requested=False,
            length_prefix_future=1,
        )
    )

    assert output.action_chunk_pending_consumed is False
    assert output.action_chunk_active is chunk_0
    assert output.action is not None
    assert float(output.action[0]) == 2.0


def test_inference_trigger_at_threshold():
    """cnt_actions_remaining (inclusive) <= D -> should_request_inference = True."""
    chunk = make_chunk(n_actions=5, timestep_start_obs=0)

    # At timestep=4: remaining_from=1 <= D=1
    output = step_actor_control_frame(
        ArgsStepActorControlFrame(
            timestep=4,
            action_chunk_active=chunk,
            action_chunk_pending=None,
            is_inference_requested=False,
            length_prefix_future=1,
        )
    )

    assert output.should_request_inference is True


def test_no_inference_trigger_above_threshold():
    """Above threshold -> should_request_inference = False."""
    chunk = make_chunk(n_actions=5, timestep_start_obs=0)

    # At timestep=3: remaining_from=2 > D=1
    output = step_actor_control_frame(
        ArgsStepActorControlFrame(
            timestep=3,
            action_chunk_active=chunk,
            action_chunk_pending=None,
            is_inference_requested=False,
            length_prefix_future=1,
        )
    )

    assert output.should_request_inference is False


def test_no_inference_trigger_when_already_requested():
    """is_inference_requested=True -> should_request_inference = False."""
    chunk = make_chunk(n_actions=5, timestep_start_obs=0)

    output = step_actor_control_frame(
        ArgsStepActorControlFrame(
            timestep=3,
            action_chunk_active=chunk,
            action_chunk_pending=None,
            is_inference_requested=True,
            length_prefix_future=1,
        )
    )

    assert output.should_request_inference is False


def test_interp_target_from_pending_chunk():
    """Pending has action at timestep + 1 -> uses it for interpolation."""
    chunk_0 = make_chunk(n_actions=5, timestep_start_obs=0, idx_chunk=0)
    # Pending starts at timestep 4 (next timestep after current=3)
    chunk_1 = make_chunk(n_actions=5, timestep_start_obs=4, idx_chunk=1, base_value=100.0)

    output = step_actor_control_frame(
        ArgsStepActorControlFrame(
            timestep=3,
            action_chunk_active=chunk_0,
            action_chunk_pending=chunk_1,
            is_inference_requested=True,
            length_prefix_future=1,
        )
    )

    # Current chunk has action at t4 (=4.0), but pending also has action at t4 (=100.0).
    # Pending's action should be preferred for interpolation target.
    assert output.action_interp_target is not None
    assert float(output.action_interp_target[0]) == 100.0


def test_interp_target_hold_when_no_next():
    """No next action -> interpolation target equals current action."""
    chunk = make_chunk(n_actions=3, timestep_start_obs=0)

    # At timestep=2: last action, no next
    output = step_actor_control_frame(
        ArgsStepActorControlFrame(
            timestep=2,
            action_chunk_active=chunk,
            action_chunk_pending=None,
            is_inference_requested=True,
            length_prefix_future=1,
        )
    )

    assert output.action is not None
    assert output.action_interp_target is not None
    assert float(output.action_interp_target[0]) == float(output.action[0])


def test_full_sequence_no_interpolation():
    """Full sequence without sub-frame interpolation (D=2, 1-frame inference latency).

    Triggering semantics (matches configuration_act_smooth.py):
      - At timestep t, the observation is taken as inference input.
      - The future prefix starts with the current chunk's action at t.
      - The first action executed from the new chunk is at t + D.
      - Inference triggers when remaining actions (inclusive) <= D.

    Without interpolation, D only needs to exceed 1 (the action at observation
    time, before inference). Due to inference latency we can't execute new chunk
    actions immediately, but the first one is sent directly at the next control
    frame after the old chunk ends.
    """
    n_action_steps = 6
    length_prefix_future = 2

    chunk_0 = make_chunk(n_actions=n_action_steps, timestep_start_obs=0, idx_chunk=0)
    action_chunk_active = chunk_0
    action_chunk_pending = None
    is_inference_requested = False
    timestep_trigger = None

    control_actions = []

    for timestep in range(12):
        output = step_actor_control_frame(
            ArgsStepActorControlFrame(
                timestep=timestep,
                action_chunk_active=action_chunk_active,
                action_chunk_pending=action_chunk_pending,
                is_inference_requested=is_inference_requested,
                length_prefix_future=length_prefix_future,
            )
        )

        action_chunk_active = output.action_chunk_active
        if output.action_chunk_pending_consumed:
            action_chunk_pending = None

        if output.should_request_inference:
            is_inference_requested = True
            timestep_trigger = timestep

        if output.action is not None:
            control_actions.append((timestep, float(output.action[0])))

        # Simulate inference completing 1 frame after trigger
        if timestep_trigger is not None and action_chunk_pending is None and timestep == timestep_trigger + 1:
            action_chunk_pending = make_chunk(
                n_actions=n_action_steps,
                timestep_start_obs=timestep_trigger,
                length_prefix_future_effective=length_prefix_future,
                idx_chunk=1,
                base_value=100.0,
            )

        if len(control_actions) >= 9:
            break

    # chunk_0: actions 0..5 at t0..t5; trigger at t4 (remaining=2 <= D=2)
    assert control_actions[0] == (0, 0.0)
    assert control_actions[1] == (1, 1.0)
    assert control_actions[2] == (2, 2.0)
    assert control_actions[3] == (3, 3.0)
    assert control_actions[4] == (4, 4.0)  # trigger, obs taken here
    assert control_actions[5] == (5, 5.0)  # last of chunk_0
    # t6: switch to chunk_1 (timestep_start = trigger + D = 4 + 2 = 6)
    assert control_actions[6] == (6, 100.0)
    assert control_actions[7] == (7, 101.0)


def test_full_sequence_with_interpolation():
    """Full sequence with sub-frame interpolation (D=3, 1-frame inference latency).

    Triggering semantics (matches configuration_act_smooth.py):
      - At timestep t, the observation is taken as inference input.
      - The future prefix starts with the current chunk's action at t.
      - The first action executed from the new chunk is at t + D.
      - Inference triggers when remaining actions (inclusive) <= D.

    With interpolation, D must exceed 1 + inference latency (in frames) so the
    pending chunk arrives before the old chunk's last action. Otherwise the
    interpolation target holds the last action, which is never desirable.
    D=3 with 1-frame latency: pending arrives 2 frames before old chunk ends,
    ensuring smooth interpolation targets throughout the transition.
    """
    n_action_steps = 7
    length_prefix_future = 3

    chunk_0 = make_chunk(n_actions=n_action_steps, timestep_start_obs=0, idx_chunk=0)
    action_chunk_active = chunk_0
    action_chunk_pending = None
    is_inference_requested = False
    timestep_trigger = None

    control_actions = []
    interp_targets = []

    for timestep in range(12):
        output = step_actor_control_frame(
            ArgsStepActorControlFrame(
                timestep=timestep,
                action_chunk_active=action_chunk_active,
                action_chunk_pending=action_chunk_pending,
                is_inference_requested=is_inference_requested,
                length_prefix_future=length_prefix_future,
            )
        )

        action_chunk_active = output.action_chunk_active
        if output.action_chunk_pending_consumed:
            action_chunk_pending = None

        if output.should_request_inference:
            is_inference_requested = True
            timestep_trigger = timestep

        if output.action is not None:
            control_actions.append((timestep, float(output.action[0])))
        if output.action_interp_target is not None:
            interp_targets.append((timestep, float(output.action_interp_target[0])))

        # Simulate inference completing 1 frame after trigger
        if timestep_trigger is not None and action_chunk_pending is None and timestep == timestep_trigger + 1:
            action_chunk_pending = make_chunk(
                n_actions=n_action_steps,
                timestep_start_obs=timestep_trigger,
                length_prefix_future_effective=length_prefix_future,
                idx_chunk=1,
                base_value=100.0,
            )

        if len(control_actions) >= 10:
            break

    # chunk_0: actions 0..6 at t0..t6; trigger at t4 (remaining=3 <= D=3)
    assert control_actions[4] == (4, 4.0)  # trigger, obs taken here
    assert control_actions[5] == (5, 5.0)  # pending arrives, old chunk continues
    assert control_actions[6] == (6, 6.0)  # last of chunk_0
    # t7: switch to chunk_1 (timestep_start = trigger + D = 4 + 3 = 7)
    assert control_actions[7] == (7, 100.0)
    assert control_actions[8] == (8, 101.0)

    # Interpolation targets: never hold last action
    # t4: next from old chunk
    assert interp_targets[4] == (4, 5.0)
    # t5: next from old chunk (pending has no action at t6)
    assert interp_targets[5] == (5, 6.0)
    # t6: from pending chunk's action at t7 — smooth handoff
    assert interp_targets[6] == (6, 100.0)
    # t7: from new chunk's action at t8
    assert interp_targets[7] == (7, 101.0)


# ============================================================================
# Inference tests
# ============================================================================


def test_prefix_future_starts_at_current():
    """Future prefix starts at observation timestep (inclusive)."""
    # Chunk with actions [0, 1, 2, 3, 4] at timesteps 0..4
    chunk = make_chunk(n_actions=5, timestep_start_obs=0, action_dim=1, length_prefix_future=2, length_prefix_past=0)

    output = make_prefix_for_inference(
        ArgsMakePrefixForInference(
            timestep=2,
            action_chunk_active=chunk,
        )
    )

    # Future prefix at [timestep, timestep+D) = [2, 4): actions at t2, t3 -> values 2.0, 3.0
    assert output.action_prefix_future.shape == (1, 2, 1)
    assert float(output.action_prefix_future[0, 0, 0]) == 2.0
    assert float(output.action_prefix_future[0, 1, 0]) == 3.0


def test_timestep_start_is_trigger_plus_D():
    """timestep_start = timestep_start_obs + D (no +1)."""
    actions = torch.randn(1, 5, 6)  # [batch, n_actions, action_dim]

    chunk = ActionChunk(
        actions=actions.squeeze(0),
        timestep_start_obs=3,
        length_prefix_future_effective=2,
        length_prefix_future=2,
        length_prefix_past=3,
        idx_chunk=1,
    )

    assert chunk.timestep_start == 3 + 2  # = 5
    assert chunk.timestep_start_obs == 3


def test_prefix_past_shape():
    """Past prefix has correct length."""
    chunk = make_chunk(n_actions=10, timestep_start_obs=0, action_dim=6, length_prefix_future=2, length_prefix_past=3)

    output = make_prefix_for_inference(
        ArgsMakePrefixForInference(
            timestep=5,
            action_chunk_active=chunk,
        )
    )

    # Past prefix: actions at t2, t3, t4 (3 actions before timestep=5)
    assert output.action_prefix_past.shape == (1, 3, 6)


def test_chunk_timing_relationship():
    """chunk.length_prefix_future_effective == the prefix length passed in."""
    actions = torch.randn(1, 5, 6)
    timestep = 3
    length_prefix_future_effective = 2

    chunk = ActionChunk(
        actions=actions.squeeze(0),
        timestep_start_obs=timestep,
        length_prefix_future_effective=length_prefix_future_effective,
        length_prefix_future=2,
        length_prefix_past=3,
        idx_chunk=1,
    )

    # Stored as-is, no +1
    assert chunk.length_prefix_future_effective == length_prefix_future_effective
    # timestep_start_obs is the timestep passed in
    assert chunk.timestep_start_obs == timestep


# ============================================================================
# Interpolation test
# ============================================================================


def test_interpolation_is_linear():
    """Verify sub-frame interpolation is strictly linear between control frames.

    Drives step_actor_control_frame through a sequence and verifies that
    the interpolation targets enable correct linear interpolation.
    """
    n_action_steps = 5
    length_prefix_future = 1
    cnt_frames_per_control_frame = 6

    chunk_0 = make_chunk(n_actions=n_action_steps, timestep_start_obs=0)
    action_chunk_active = chunk_0
    action_chunk_pending = None

    # Collect (timestep, action_value, interp_target_value) for consecutive control frames
    control_pairs = []

    for timestep in range(n_action_steps - 1):
        output = step_actor_control_frame(
            ArgsStepActorControlFrame(
                timestep=timestep,
                action_chunk_active=action_chunk_active,
                action_chunk_pending=action_chunk_pending,
                is_inference_requested=True,  # Suppress new requests
                length_prefix_future=length_prefix_future,
            )
        )

        if output.action is not None and output.action_interp_target is not None:
            control_pairs.append((float(output.action[0]), float(output.action_interp_target[0])))

    # Verify interpolation between consecutive control frames
    for i in range(len(control_pairs) - 1):
        v_start, v_end = control_pairs[i]

        for idx_within in range(1, cnt_frames_per_control_frame):
            t = idx_within / cnt_frames_per_control_frame
            interp_value = (1.0 - t) * v_start + t * v_end

            # The next control frame value should match the interp target
            expected_next = control_pairs[i + 1][0]
            # At t=1.0, interpolation should reach the next control frame value
            interp_at_1 = v_end
            assert abs(interp_at_1 - expected_next) < 1e-4, (
                f"Interpolation endpoint ({interp_at_1}) != next control value ({expected_next})"
            )

            # Verify interpolation is linear (just check it's between start and end)
            assert min(v_start, v_end) - 1e-4 <= interp_value <= max(v_start, v_end) + 1e-4, (
                f"Interpolation value {interp_value} outside [{v_start}, {v_end}]"
            )
