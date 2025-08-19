# Formula Student Racing Line Optimizer

Plan, optimize, and simulate racing lines for cone‑delineated Formula Student tracks.

## Table of Contents
- [Overview](#overview)
- [Core Components](#core-components)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Data Format](#data-format)
- [Examples & Recipes](#examples--recipes)
- [API Quick Reference](#api-quick-reference)
- [Architecture](#architecture)
- [Performance & Limits](#performance--limits)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Testing & QA](#testing--qa)
- [Optional next tweaks](#optional-next-tweaks)
- [Security](#security)
- [FAQ](#faq)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
The toolkit turns cone positions into drivable paths and evaluates control strategies in a lightweight simulator.

### Highlights
- Hungarian or greedy cone pairing with geometric ordering
- Laplacian/spline smoothing and convex racing‑line optimization
- Kinematic bicycle model with Pure Pursuit, Stanley, and MPPI controllers
- Matplotlib UI, telemetry logging, and headless rendering
- Optional ROS 2 node for real‑time integrations

## Core Components
- `fmsim/utils.py` — I/O helpers (`load_cones_json()`, `car_triangle()`)
- `fmsim/planner.py` — pairing (greedy/hungarian), smoothing (`laplacian_smooth()`, `spline_smooth()`), optimization (`optimization_based_racing_line()`), curvature
- `fmsim/models.py` — `VehicleParams`, `BicycleKinematic`, controllers (`pure_pursuit_control`, `stanley_control`, `MPPIController`), `TelemetryLogger`
- `fmsim/ros2_interface.py` — optional ROS 2 node (cones in, path/telemetry out)
- `scripts/sim_headless.py` — off‑screen renderer producing `artifacts/track.png`
- `main.py` — CLI and demo entrypoint

Dependency fallbacks:
- If SciPy unavailable ➜ Hungarian pairing falls back to greedy
- If `UnivariateSpline` unavailable ➜ `spline_smooth()` falls back to linear/Laplacian
- If CVXPY/solver fails ➜ `optimization_based_racing_line()` falls back to midline

## Quickstart

### Prerequisites
- Python 3.7+ (3.10 recommended; CI-tested)
- Optional: SciPy (pairing speedups), CVXPY (optimal racing line), ROS 2 (`rclpy`) for the node

### Install
```bash
git clone https://github.com/<user>/fs-racing-line-optimizer.git
cd fs-racing-line-optimizer
python3 -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

### Hello World
```python
import numpy as np
from fmsim.utils import load_cones_json
from fmsim.planner import pair_cones_to_midline, laplacian_smooth
from fmsim.models import VehicleParams, BicycleKinematic, pure_pursuit_control

left, right = load_cones_json("data/sample_cones.json")
mid  = pair_cones_to_midline(left, right, method="hungarian")
path = laplacian_smooth(mid, corridor=(left, right))

car   = BicycleKinematic(VehicleParams())
state = np.array([path[0,0], path[0,1], 0.0, 5.0])
steer, _ = pure_pursuit_control(state, path)
```

## Usage

### CLI
```bash
python main.py --cones data/sample_cones.json \
               --pairing hungarian --smoothing spline \
               --controller stanley --noise --latency
```

### Flags
- `--pairing`: greedy or hungarian cone pairing
- `--smoothing`: laplacian, spline, or optimization
- `--controller`: pure_pursuit, stanley, or mppi
- `--noise`: enable IMU noise
- `--latency`: inject control delay
- `--demo`: run a feature showcase

- Recommended baseline: `--pairing hungarian --smoothing spline --controller stanley` (defaults)
  Defaults: pairing=hungarian, smoothing=spline, controller=stanley

### Library
```python
from fmsim.planner import optimization_based_racing_line
opt_line = optimization_based_racing_line(left, right, num_points=80)
```

### Configuration
- See Data Format for cones JSON schema.
- Headless render: `scripts/sim_headless.py --cones data/sample_cones.json --out artifacts` or `make headless`.

## Data Format
Cones JSON schema:
```json
{
  "left": [[x, y], ...],
  "right": [[x, y], ...]
}
```
Units: meters in a common 2D plane. Left/right arrays may differ in length.
Type/shape: float arrays of shape (N,2).
Edge cases: if either side is empty, midline falls back to an ordered polyline of all cones.

## Examples & Recipes

Compute curvature:
```python
from fmsim.planner import curvature_geometric
kappa = curvature_geometric(path)
```

- Log telemetry to CSV with `TelemetryLogger`.
- Swap controllers in the main loop to compare steering strategies.
- Use `pair_cones_to_midline(..., method="greedy")` for minimal SciPy setups.

## API Quick Reference
- Pairing: `pair_cones_to_midline(left, right, method='greedy'|'hungarian') -> (M,2)`
- Smoothing: `laplacian_smooth(path, alpha=..., iters=..., corridor=(L,R))`; `spline_smooth(path, corridor=(L,R))`
- Optimization: `optimization_based_racing_line(left, right, num_points=...) -> (M,2)`
- Curvature: `curvature_geometric(path) -> (len(path),)`
- Vehicle: `BicycleKinematic(VehicleParams(...)).step(state, (steer, accel), dt)`
- Controllers: `pure_pursuit_control(state, path)`; `stanley_control(state, path)`; `MPPIController(...).control(state, path, dt)`
- Telemetry: `TelemetryLogger().log(t, state, control, path)`; `save_to_file(file)`

## Architecture
```mermaid
graph TD
    A[Cones JSON] --> B[Planner (pair + smooth)]
    B --> C[Path]
    C --> D[Controller]
    D --> E[BicycleKinematic]
    E --> C
    E --> F[Telemetry]
    E --> G[UI / Headless]
```

See `docs/design_notes.md` for algorithm notes.

## Performance & Limits
- Greedy pairing O(n²); Hungarian O(n³) on the cost matrix (uses SciPy; else greedy).
- Spline smoothing uses SciPy `UnivariateSpline` (fallback to linear/Laplacian). Optimization uses CVXPY+OSQP; runtime grows with `num_points`.
- Simulation uses dt=0.03 s (~33 Hz). Heavier controllers (e.g., MPPI) can reduce UI frame rate.

## Roadmap
- Hungarian cone pairing (completed)
- Spline smoothing with corridor constraints
- Deterministic IMU noise & telemetry logging
- ROS 2 topic interface (see `docs/ros2_topics.md`)
- More in `docs/roadmap.md`.

## Contributing
```bash
pip install -e .[dev]
flake8 fmsim main.py scripts tests
pytest
```

Submit a PR with clear description and tests.

Make targets:
```bash
make lint
make test
make check
make run
make headless
```

## Security
Report vulnerabilities via GitHub Issues or directly to the maintainers. Automated dependency checks run in CI.

## FAQ
- **Install fails on system Python** — Create a venv (`python -m venv venv`) to bypass PEP 668 restrictions.
- **SciPy/CVXPY import errors** — These are optional; install them or switch to greedy pairing and Laplacian smoothing.
- **Matplotlib complains about display** — Use `scripts/sim_headless.py` for off‑screen rendering.
- **Simulation drifts or oscillates** — Try the Stanley controller or reduce lookahead.
- **Where’s the sample track?** — `data/sample_cones.json` contains an S‑curve and chicane track.

## Optional next tweaks
- pyproject: Consider bumping `requires-python` to ">=3.10" to match CI.
- Polish: Add CI/coverage badges and embed `artifacts/track.png`.

## License
Distributed under the MIT License. See `LICENSE`.

## Acknowledgements
Inspired by a conversation with a Member of Formula Manipal