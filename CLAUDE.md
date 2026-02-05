# CLAUDE.md

This file provides guidance for Claude Code working in this repository.

## Project Overview

Python 3.11+ monorepo for the ACTSmooth policy and its evaluation on the SO101 arm.

## Environment Setup

This project uses **nix + uv** (not conda as LeRobot docs suggest).

To enter the dev environment:

```bash
nix develop
uv sync
```

Remember to source the virtual environment at `.venv` when running Python scripts.

## LeRobot Setup

This project uses the LeRobot library for robotics data collection with the SO101 robot arm.

LeRobot is installed from a **pinned git commit** (not PyPI) because the latest PyPI version (0.4.2) is outdated. See `pyproject.toml` for the exact commit hash.

IMPORTANT: When you need to understand how LeRobot works internally, explore its source files in

```
.venv/lib/python3.10/site-packages/lerobot/
```

## Code Naming

Follow a noun-first approach for variables and constants, for example `idChat` instead of `chatId`. Follow a verb-first approach for functions, for example `computeContentFromTree` or `fetchDataUser`. Notice that we still follow the noun-first approach, it's `fetchDataUser` instead of `fetchUserData`. Plural applies to the noun, for example for multiple user ids use `idsUser` instead of `idsUsers` or `idUsers`.

**Noun-first patterns:**

- Timesteps: `timestep_obs`, `timestep_action` (not `obs_timestep`)
- Observations: `dict_obs` (raw dict), `proprio_obs` (proprioception), `dict_obs_current`
- Actions: `action_chunk_pending`, `action_chunk_active`, `tensor_action`
- Events: `event_shutdown`, `event_inference_requested`
- Counts: `cnt_actions`, `cnt_total_actions` (not `action_count` or `n_actions`)
- Lengths: `length_prefix_past`, `length_chunk` (not `n_prefix_past`)
- Indices: `idx_chunk`, `idx_frame` (not `chunk_idx`)
- Threads: `thread_inference`, `thread_actor`
- Paths: `path_recording`, `path_output`
- Trackers: `tracker_latency`
- Booleans: `is_inference_running`, `is_control_frame` (use `is_` prefix for state flags)

**Exception: Library Interface Consistency**

When interfacing with external libraries (e.g., LeRobot, PyTorch, ACTRelativeRTC), preserve the library's naming conventions for:

- Function parameters that match library APIs (e.g., `delay`, `action_prefix` for ACTRelativeRTC)
- Dict keys that follow library conventions (e.g., `observation.state`, `observation.images.*` for LeRobot)
- Variables passed directly to library functions (e.g., `robot_action` for `robot.send_action()`)
- Standard framework naming (e.g., `device` for PyTorch, `policy`, `preprocessor` for LeRobot)

The goal is: use noun-first naming for internal code, but maintain consistency at library boundaries to reduce cognitive friction when reading code that interfaces with external APIs.

**File/folder prefixes:**

- `bak` prefix/suffix: backup files and folders, ignore unless explicitly asked
- `tmp` prefix/suffix: temporary files and folders, ignore unless explicitly asked
- `zxtra` prefix/suffix: extra files and folders (work-in-progress or uncertain), generally ignore

Align the caller's variable names with the callee's parameter names to reduce cognitive load and make data flow explicit. The variable names outside a function must match the internal parameters exactly. The exception is when handling generic sequences or iterative data, where indexed suffixes (like `_1, _2, _3`) are used to represent distinct instances of the same conceptual type.

## Code Quality

- **Formatting**: Uses ruff for code formatting (auto-runs via Claude Code hook on file save)
- **Linting**: Uses ruff for linting
- **Type Checking**: Disabled (no mypy/pyright/ty)

Run manually if needed:

```bash
uv run ruff format .
uv run ruff check .
```
