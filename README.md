# Newton-Raphson Flow for PX4-ROS2 Deployment
![Status](https://img.shields.io/badge/Status-Hardware_Validated-blue)
[![ROS 2 Compatible](https://img.shields.io/badge/ROS%202-Humble-blue)](https://docs.ros.org/en/humble/index.html)
[![PX4 Compatible](https://img.shields.io/badge/PX4-Autopilot-pink)](https://github.com/PX4/PX4-Autopilot)
[![evannsmc.com](https://img.shields.io/badge/evannsmc.com-Project%20Page-blue)](https://www.evannsmc.com/projects/nr-flow)

This package is the culmination of 3 papers on Newton-Raphson Flow for Quadrotor Control.

This package allows for fast, accuate, and computationally efficient control via the Newton-Raphson Flow (NR Flow) controller developed by Dr. Yorai Wardi and others. We introduce integral CBFs (I-CBFs) to smoothly limit control actuation.

The NR Flow controller is an integral-based control strategy based on a continuous time flow-version of the known newton-raphson iterative algorithm for finding the zeros of functions. It has been shown to have desirable theoretical properties in previous work, including known tracking error bounds, and we show in our hardware implementations that it compares favorably to the native control stack of PX4 Autopilot, as well as NMPC. Notably, it outperforms NMPC in terms of speed and computational efficiency (measured by joules of energy expended by the CPU), and on complex trajectories it may even outperform NMPC due to computational constraints. This is an ideal controller when facing on-board computational limitations. In particular, we test and deploy this on an on-board Raspberry Pi 4 Model B on a Holybro x500V2 quadrotor and we compare it against the NMPC controller available in my [`NMPC_PX4`](https://github.com/evannsm/NMPC_PX4) package.

## Key Features

- **Newton-Raphson control law** — iterative inversion of the system Jacobian for feedback linearization
- **Integral CBF safety constraints** — optional barrier functions to enforce input constraints (enabled by default)
- **JAX JIT-compiled** — all control computations are JIT-compiled for real-time performance
- **PX4 integration** — publishes attitude setpoints and offboard commands via `px4_msgs`
- **Structured logging** — optional CSV logging via ros2_logger with automatic analysis notebook generation

## Controller Profiles

The standard Python node now exposes two explicit Newton-Raphson profiles via
`--nr-profile`:

| Profile | Lookahead | Predictor | Iterations | `alpha` | Integral action |
| ------- | --------- | --------- | ---------- | ------- | --------------- |
| `baseline` | `1.2 s` | ZOH | `1` | `[50, 60, 60, 60]` | Disabled |
| `workshop` | `0.8 s` | FOH | `2` | `[45, 55, 55, 45]` | Enabled with bounded anti-windup |

`baseline` preserves the current controller structure for direct comparisons.
`workshop` is the validated structural improvement profile: shorter lookahead,
first-order-hold prediction, bounded integral error injection, and two damped
Newton updates per 100 Hz control cycle.

For the measured Python comparison on April 1, 2026, see:

- `docs/newton_raphson_workshop_profiles.qmd`

## Usage

```bash
# Source your ROS 2 workspace
source install/setup.bash

# Fly a circle in simulation
ros2 run newton_raphson_px4 run_node --platform sim --trajectory circle_horz

# Fly a helix on hardware with logging
ros2 run newton_raphson_px4 run_node --platform hw --trajectory helix --log

# Hover mode 3, double speed, with yaw spin
ros2 run newton_raphson_px4 run_node --platform sim --trajectory hover --hover-mode 3 --double-speed --spin

# fig8_contraction with feedforward, logged with _ff marker in filename
ros2 run newton_raphson_px4 run_node --platform sim --trajectory fig8_contraction --ff --log
# -> logs to: sim_nr_std_fig8_contraction_ff_1x.csv

# Run the validated workshop profile
ros2 run newton_raphson_px4 run_node --platform sim --trajectory fig8_horz --nr-profile workshop --log
```

### CLI Options

| Flag                                            | Description                                                    |
| ----------------------------------------------- | -------------------------------------------------------------- |
| `--platform {sim,hw}`                           | Target platform (required)                                     |
| `--trajectory {hover,yaw_only,circle_horz,...}` | Trajectory type (required)                                     |
| `--hover-mode {1..8}`                           | Hover sub-mode (1-4 for hardware)                              |
| `--log`                                         | Enable CSV data logging                                        |
| `--log-file NAME`                               | Custom log filename                                            |
| `--double-speed`                                | 2x trajectory speed                                            |
| `--short`                                       | Short variant (fig8_vert)                                      |
| `--spin`                                        | Enable yaw rotation                                            |
| `--flight-period SEC`                           | Custom flight duration                                         |
| `--ff`                                          | Mark log filename with `_ff` (only valid with `fig8_contraction`) |
| `--nr-profile {baseline,workshop}`              | Select the Newton-Raphson profile                              |

## Feedforward for `fig8_contraction`

When the `fig8_contraction` trajectory is selected, the node computes a differential-flatness feedforward at each control step using the same approach as the contraction controller (`flat_to_x_u` from `quad_trajectories`).

**How it works:**

1. The flat output `[px, py, pz, psi](t)` is differentiated twice via `jax.jacfwd` to recover velocity and acceleration.
2. From acceleration, the feedforward specific thrust `f` and Euler angles `[phi, th, psi]` are computed analytically (flat-output inversion).
3. A third differentiation gives `u_ff = [df, dphi, dth, dpsi]` — the rates of thrust, roll, pitch, and yaw.
4. The angular rate feedforward `u_ff[1:4] = [dphi, dth, dpsi]` is **added directly to the NR control output** `[roll_rate, pitch_rate, yaw_rate]`, providing a baseline that the NR feedback corrects around rather than building up from zero.

The thrust component `u_ff[0] = df` is not added to the NR thrust output (they live in different units: `df` is in m/s³, NR thrust is in N), so NR handles thrust authority through position tracking as normal.

## Dependencies

- [quad_trajectories](https://github.com/evannsm/quad_trajectories) — trajectory definitions
- [quad_platforms](https://github.com/evannsm/quad_platforms) — platform abstraction
- [ros2_logger](https://github.com/evannsm/ROS2Logger) — experiment logging
- [px4_msgs](https://github.com/PX4/px4_msgs) — PX4 ROS 2 message definitions
- JAX / jaxlib

## Package Structure

```
newton_raphson_px4/
├── newton_raphson_px4/
│   ├── run_node.py              # CLI entry point and argument parsing
│   └── ros2px4_node.py          # ROS 2 node (subscriptions, publishers, control loop)
└── newton_raphson_px4_utils/
    ├── controller/
    │   ├── newton_raphson_px4.py       # Newton-Raphson control law
    │   └── nr_utils.py          # Dynamics, Jacobians, CBF functions
    ├── px4_utils/               # PX4 interface and flight phase management
    ├── transformations/         # Yaw adjustment utilities
    ├── main_utils.py            # Helper functions
    └── jax_utils.py             # JAX configuration
```

## Installation

```bash
# Inside a ROS 2 workspace src/ directory
git clone git@github.com:evannsm/newton_raphson_px4.git
cd .. && colcon build --symlink-install
```

## License
MIT

## Website

This project is part of the [evannsmc open-source portfolio](https://www.evannsmc.com/projects).

- [Project page](https://www.evannsmc.com/projects/nr-flow)

# Papers and their repositories:
American Control Conference 2024 - [see paper here](https://coogan.ece.gatech.edu/papers/pdf/cuadrado2024tracking.pdf)  
[Personal Version of Repository](https://github.com/evannsm/MoralesCuadrado_ACC2024)  
[Official FACTSLab Repository](https://github.com/gtfactslab/MoralesCuadrado_Llanes_ACC2024)  

Transactions on Control Systems Technology 2025 - [see paper here](https://arxiv.org/abs/2508.14185)  
[Personal Version of Repository](https://github.com/evannsm/MoralesCuadrado_Baird_TCST2025)  
[Official FACTSLab Repository](https://github.com/gtfactslab/Baird_MoralesCuadrado_TRO_2025)  

Transactions on Robotics 2025  
[Personal Version of Repository](https://github.com/evannsm/MoralesCuadrado_Baird_TCST2025)  
[Official FACTSLab Repository](https://github.com/gtfactslab/MoralesCuadrado_Baird_TCST2025)  
