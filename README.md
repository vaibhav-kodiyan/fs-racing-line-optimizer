# Formula Student Racing Line Optimizer

A Python-based racing line optimization system for Formula Student/SAE competition, featuring advanced path planning and vehicle control algorithms.

## Features

- **Path Planning**
  - Robust cone pairing algorithm for track boundary detection
  - Laplacian smoothing with corridor constraints
  - Curvature-aware path optimization
  - Support for complex track layouts

- **Vehicle Control**
  - Pure Pursuit path following controller
  - Kinematic bicycle model for vehicle dynamics
  - Adaptive lookahead distance based on speed
  - Robust to sensor noise and imperfect path data

- **Performance**
  - Efficient algorithms for real-time operation
  - Optimized for embedded systems
  - Minimal external dependencies

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fs-racing-line-optimizer.git
   cd fs-racing-line-optimizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Path Planning

```python
from fmsim.planner import pair_cones_to_midline, laplacian_smooth, curvature_discrete
import numpy as np

# Example cone positions
left_cones = np.array([[0, 1], [1, 1.1], [2, 0.9], [3, 1.0]])
right_cones = np.array([[0, -1], [1, -0.9], [2, -1.1], [3, -1.0]])

# Generate midline path
midline = pair_cones_to_midline(left_cones, right_cones)
smoothed_path = laplacian_smooth(midline, corridor=(left_cones, right_cones))
curvature = curvature_discrete(smoothed_path)
```

### Vehicle Control

```python
from fmsim.models import VehicleParams, BicycleKinematic, pure_pursuit_control
import numpy as np

# Initialize vehicle model
params = VehicleParams(wheelbase=1.6, max_steer=np.deg2rad(35))
vehicle = BicycleKinematic(params)

# Initial state: [x, y, yaw, velocity]
state = np.array([0.0, 0.0, 0.0, 5.0])
path = np.array([[0, 0], [10, 0], [20, 5], [30, 10]])

# Control loop
for _ in range(100):
    # Calculate steering command
    steer, _ = pure_pursuit_control(state, path)
    
    # Update vehicle state (simplified)
    state = vehicle.step(state, (steer, 0.0), dt=0.1)
```

## Improvements in v2.0

### Core Enhancements
- **Robust Cone Pairing**: Improved handling of uneven cone distributions
- **Accurate Curvature Calculation**: Proper geometric curvature instead of simple second differences
- **Stable Control**: Better handling of edge cases in the Pure Pursuit controller
- **Numerical Stability**: Improved handling of edge cases and numerical precision

### Performance Optimizations
- KD-tree based nearest neighbor search for faster cone pairing
- Early termination in smoothing algorithm when convergence is reached
- More efficient path ordering algorithm

### Code Quality
- Comprehensive docstrings and type hints
- Better error handling and input validation
- More maintainable code structure

## Testing

Run the test suite with:

```bash
pytest tests/
```

## Dependencies

- Python 3.7+
- NumPy
- SciPy (for KD-tree in cone pairing)
- Matplotlib (for examples and visualization)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Virtual Environment (Recommended)

To avoid system Python restrictions (PEP 668), use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Development Install

Install the library in editable mode so changes reflect immediately:

```bash
pip install -e .
pip install -e ".[dev]"
```

## Run the Simulator

```bash
python main.py
```

The app loads cones from `data/sample_cones.json`, builds a midline via `pair_cones_to_midline()` and `laplacian_smooth()`, and simulates a car controlled by `pure_pursuit_control()` using the `BicycleKinematic` model.

### Keyboard Controls

- Space: Pause/Resume
- Esc: Quit

## Sample Track

The default `data/sample_cones.json` now includes:

- Tight S-curves and a chicane
- Varying track widths (narrowing and widening)
- Longer layout (~125 m) to stress-test controller and smoothing

## Notable Changes (v2.0)

- Pure Pursuit: speed-aware lookahead, vehicle-frame targeting, safer index selection
- Bicycle model: yaw normalization and velocity update stability
- Planner: KD-tree pairing, improved polyline ordering, convergence-based smoothing
- Curvature: geometric curvature via derivatives for better fidelity
- UI: fixed Matplotlib cache warning by setting `save_count` and `cache_frame_data`
- Packaging: added `pyproject.toml` and `setup.py`; `pip install -e .` works

## Troubleshooting

- If `pip install -r requirements.txt` fails with externally-managed environment, use the Virtual Environment steps above.
- If `pytest` is not found, install via your venv or system package manager, or run `pip install -e ".[dev]"`.

## Command-line Options

The simulator accepts the following argument:

```bash
python main.py --cones PATH/TO/CONES.json
```

- `--cones`: Path to a cones JSON file (defaults to `data/sample_cones.json`).

## Cones JSON Format

The expected schema is:

```json
{
  "left":  [[x0, y0], [x1, y1], ...],
  "right": [[x0, y0], [x1, y1], ...]
}
```

- Coordinates are in meters in a common 2D plane.
- Arrays can be of different lengths; the planner will pair greedily and order them.

## Repository Structure

- `fmsim/models.py`: Vehicle parameters, kinematic bicycle, pure-pursuit controller
- `fmsim/planner.py`: Cone pairing, polyline ordering, smoothing, curvature
- `fmsim/ui.py`: Matplotlib-based visualization and animation
- `fmsim/utils.py`: I/O utilities (e.g., `load_cones_json`)
- `data/sample_cones.json`: Example track cones
- `tests/`: Unit tests for planner and controller
- `main.py`: Entry-point to run the simulator

## Changelog

### v2.0

- Controller: speed-aware lookahead, vehicle-frame geometry, robust target selection
- Model: yaw normalization, velocity non-negativity, steering clamp
- Planner: KD-tree pairing, convergence-aware smoothing, geometric curvature
- UI: animation cache warning fixed, minor layout polish
- Packaging: editable install via `pyproject.toml` and `setup.py`
- Data: challenging sample track with S-curves and chicanes
- Docs: expanded README with setup, run, controls, and troubleshooting

## Known Limitations / Future Work

- Pure pursuit can cut apexes aggressively on very tight S-curves; consider Stanley/MPPI
- Smoothing is Laplacian-based; an optimization-based corridor solver could yield faster lines
- No longitudinal tire/traction limits; current accel is a simple proportional target
- No noise/latency in the loop; add for robustness testing