# Newton-Raphson Standard Controller Commands

## Build Package

```bash
cd ~/ws_clean_traj
colcon build --packages-select quad_platforms quad_trajectories newton_raphson_px4
source install/setup.bash
```

## Run Commands

### Simulation

```bash
# Hover (requires --hover-mode)
ros2 run newton_raphson_px4 run_node --platform sim --trajectory hover --hover-mode 1

# Circle Horizontal
ros2 run newton_raphson_px4 run_node --platform sim --trajectory circle_horz
ros2 run newton_raphson_px4 run_node --platform sim --trajectory circle_horz --double-speed
ros2 run newton_raphson_px4 run_node --platform sim --trajectory circle_horz --spin
ros2 run newton_raphson_px4 run_node --platform sim --trajectory circle_horz --double-speed --spin

# Circle Vertical
ros2 run newton_raphson_px4 run_node --platform sim --trajectory circle_vert
ros2 run newton_raphson_px4 run_node --platform sim --trajectory circle_vert --double-speed

# Figure-8 Horizontal
ros2 run newton_raphson_px4 run_node --platform sim --trajectory fig8_horz
ros2 run newton_raphson_px4 run_node --platform sim --trajectory fig8_horz --double-speed

# Figure-8 Vertical
ros2 run newton_raphson_px4 run_node --platform sim --trajectory fig8_vert
ros2 run newton_raphson_px4 run_node --platform sim --trajectory fig8_vert --double-speed
ros2 run newton_raphson_px4 run_node --platform sim --trajectory fig8_vert --short
ros2 run newton_raphson_px4 run_node --platform sim --trajectory fig8_vert --double-speed --short

# Helix
ros2 run newton_raphson_px4 run_node --platform sim --trajectory helix
ros2 run newton_raphson_px4 run_node --platform sim --trajectory helix --double-speed
ros2 run newton_raphson_px4 run_node --platform sim --trajectory helix --spin
ros2 run newton_raphson_px4 run_node --platform sim --trajectory helix --double-speed --spin

# Sawtooth
ros2 run newton_raphson_px4 run_node --platform sim --trajectory sawtooth
ros2 run newton_raphson_px4 run_node --platform sim --trajectory sawtooth --double-speed

# Triangle
ros2 run newton_raphson_px4 run_node --platform sim --trajectory triangle
ros2 run newton_raphson_px4 run_node --platform sim --trajectory triangle --double-speed

# Yaw Only
ros2 run newton_raphson_px4 run_node --platform sim --trajectory yaw_only
ros2 run newton_raphson_px4 run_node --platform sim --trajectory yaw_only --double-speed

# Figure-8 Contraction (no feedforward marker)
ros2 run newton_raphson_px4 run_node --platform sim --trajectory f8_contraction
ros2 run newton_raphson_px4 run_node --platform sim --trajectory f8_contraction --double-speed

# Figure-8 Contraction with feedforward log marker
ros2 run newton_raphson_px4 run_node --platform sim --trajectory f8_contraction --ff
ros2 run newton_raphson_px4 run_node --platform sim --trajectory f8_contraction --ff --double-speed
```

### Hardware

```bash
# Hover (modes 1-4 only on hardware)
ros2 run newton_raphson_px4 run_node --platform hw --trajectory hover --hover-mode 1
ros2 run newton_raphson_px4 run_node --platform hw --trajectory hover --hover-mode 2
ros2 run newton_raphson_px4 run_node --platform hw --trajectory hover --hover-mode 3
ros2 run newton_raphson_px4 run_node --platform hw --trajectory hover --hover-mode 4

# Circle Horizontal
ros2 run newton_raphson_px4 run_node --platform hw --trajectory circle_horz
ros2 run newton_raphson_px4 run_node --platform hw --trajectory circle_horz --double-speed

# Helix
ros2 run newton_raphson_px4 run_node --platform hw --trajectory helix
ros2 run newton_raphson_px4 run_node --platform hw --trajectory helix --double-speed
```

## With Logging

Add `--log` to auto-generate log filename based on configuration:

```bash
# Auto-generated filename: sim_nr_std_helix_2x_spin.csv
ros2 run newton_raphson_px4 run_node --platform sim --trajectory helix --double-speed --spin --log

# Auto-generated filename: sim_nr_std_f8_contraction_ff_1x.csv
ros2 run newton_raphson_px4 run_node --platform sim --trajectory f8_contraction --ff --log

# Auto-generated filename: sim_nr_std_f8_contraction_ff_2x.csv
ros2 run newton_raphson_px4 run_node --platform sim --trajectory f8_contraction --ff --double-speed --log

# Custom filename
ros2 run newton_raphson_px4 run_node --platform sim --trajectory helix --log --log-file my_experiment
```

## With PyJoules Energy Monitoring

```bash
ros2 run newton_raphson_px4 run_node --platform sim --trajectory helix --double-speed --pyjoules
ros2 run newton_raphson_px4 run_node --platform sim --trajectory helix --double-speed --log --pyjoules
```

## Arguments Reference

| Argument | Required | Values | Description |
|----------|----------|--------|-------------|
| `--platform` | Yes | `sim`, `hw` | Platform type |
| `--trajectory` | Yes | `hover`, `yaw_only`, `circle_horz`, `circle_vert`, `fig8_horz`, `fig8_vert`, `helix`, `sawtooth`, `triangle`, `f8_contraction` | Trajectory type |
| `--hover-mode` | If hover | `1-8` (sim), `1-4` (hw) | Hover position |
| `--double-speed` | No | flag | 2x trajectory speed |
| `--short` | No | flag | Short fig8_vert variant |
| `--spin` | No | flag | Enable yaw rotation |
| `--log` | No | flag | Enable data logging |
| `--log-file` | No | string | Custom log filename |
| `--pyjoules` | No | flag | Energy monitoring |
| `--ff` | No | flag | Mark log filename with `_ff` (only valid with `f8_contraction`) |

## Difference from Enhanced Controller

The **Standard** Newton-Raphson controller uses only tracking error feedback:
```
u = last_input + ALPHA * (dgdu_inv @ error) * dt
```

The **Enhanced** controller adds feedforward from trajectory derivatives:
```
u = last_input + (dgdu_inv @ (ALPHA * error + enhanced_error_term)) * dt
```

where `enhanced_error_term = reference_rate - dgdx @ dynamics(state, input)`
