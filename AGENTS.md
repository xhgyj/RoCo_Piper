# Repository Guidelines

## Project Structure & Module Organization

Python sources live in `src/rocobrick/`. Environment setup is under `env/`, robot kinematics under `robot/`, task parsing under `task_config/`, and scripted or learned policies under `policy/`. Executable workflows are in `run/`: `demo.py` runs the expert, `collect_demos.py` records LeRobot episodes, `check_cameras.py` validates camera output, and `main.py` is the evaluation entry point. Robot USD/URDF assets belong in `robot_assets/`; task definitions are grouped by type in `tasks/`. Runtime settings are split between `config/system_config.json` and `config/user_config.json`. Add automated tests to `tests/`.

## Build, Test, and Development Commands

- `uv sync --locked` creates the Python 3.11 environment from `uv.lock`.
- `uv run bricksim ./run/demo.py` launches the scripted expert in Isaac Sim.
- `uv run bricksim ./run/check_cameras.py --output /tmp/roco-camera-check` performs a camera smoke test without leaving repository artifacts.
- `uv run bricksim ./run/collect_demos.py --episodes 10 --repo-id local/roco-piper-act --output datasets/roco_piper_act` collects demonstrations.
- `uv run pytest` runs the test suite; use `uv run pytest tests/test_piper_expert_config.py` for a focused check.
- `uv run ruff check src tests` checks formatting-independent style, imports, naming, and docstrings.

Isaac-dependent commands require an NVIDIA GPU and may need the ROS 2 bridge environment documented by Isaac Sim. Prefer headless flags supported by the target script for remote systems.

## Coding Style & Naming Conventions

Use four-space indentation, Python 3.11 syntax, and Google-style docstrings. Follow Ruff's configured `E`, `F`, `N`, `I`, and documentation rules. Use `snake_case` for functions and modules, `PascalCase` for classes, and descriptive configuration keys consistent with existing JSON. Avoid `typing.Any` and blanket `typing.cast`; define concrete types or protocols instead.

## Testing Guidelines

Tests use pytest and follow `test_*.py` / `test_*` naming. Keep pure policy and configuration checks runnable without launching Isaac Sim. For simulator changes, pair unit tests with a short BrickSim smoke run and verify success state, camera health, and saved dataset dimensions. Do not commit generated datasets, screenshots, caches, or temporary camera output.

## Commit & Pull Request Guidelines

History follows concise Conventional Commit subjects such as `feat:`, `fix:`, `perf:`, and `chore:`. Keep commits scoped and imperative. Pull requests should explain behavior changes, list exact verification commands, link relevant issues, and include representative logs or images for visual/simulation changes. Preserve unrelated working-tree changes.
