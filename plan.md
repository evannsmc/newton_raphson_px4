# Plan: revert NR-algorithm functional changes from `main`, preserve on `workshop`

Baseline reference commit: **3b0b21c** ("Align gravity constant to 9.8").
Goal: `main`'s Newton-Raphson algorithm returns to its 3b0b21c behavior. All
functional "workshop" changes live on the `workshop` branch. Cosmetic/housekeeping
changes stay on `main`.

## Decisions (confirmed with user)
- Branch name for functional work: **`workshop`**.
- `--ff` feedforward generalization (works with any trajectory): **KEEP on main**.
- PyJoules energy-log filename fix + `setup.py` tests_require removal: keep on main (non-NR).
- New request: rename README dependency display `ros2_logger` -> `ROS2Logger`.

## Tasks
- [x] Preserve all current work on `workshop` branch and push (origin/workshop @ 554f226)
- [x] Revert `newton_raphson_px4_utils/controller/newton_raphson_px4.py` to 3b0b21c (whole-file: pure workshop)
- [x] Revert `newton_raphson_px4_utils/controller/nr_utils.py` to 3b0b21c (whole-file: pure workshop; keeps GRAVITY=9.8)
- [x] Rebuild `ros2px4_node.py`: checkout 3b0b21c, re-apply ONLY the 4 ff-generalization edits
- [x] Surgically strip workshop bits from `run_node.py` (keep pyjoules fix + ff validation removal)
- [x] Edit `README.md`: restore "Control Parameters" section, drop workshop usage/CLI rows, rename ros2_logger -> ROS2Logger
- [x] Verify: no leftover `nr_profile`/`build_nr_profile`/`workshop` refs in code; python syntax check
- [x] Commit on `main` and push

## Notes / changed scope
- ff-generalization edits in ros2px4_node.py (re-applied on top of 3b0b21c):
  1. ff-compile guard: `if FIG8 and self.feedforward:` -> `if self.feedforward:`
  2. ff ctx double_speed: `False` -> `False if FIG8 else self.double_speed`
  3. flat_output: `TRAJ_REGISTRY[FIG8]` -> `TRAJ_REGISTRY[self.ref_type]`
  4. control guard: `if FIG8 and self.feedforward and self._ff_jit:` -> `if self.feedforward and self._ff_jit:`
- NOTE: `--ff` help text + README note still say "only valid with fig8_contraction".
  Left as-is to keep change focused; the validation that enforced it is removed (ff-generalization kept).
