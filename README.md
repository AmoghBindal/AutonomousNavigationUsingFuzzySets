# Fuzzy set–based autonomous navigation — implementation & analysis

Re-implementation of E. A. Rastkhiz, H. Schwartz, I. Lambadaris,
*A fuzzy set-based methodology for autonomous navigation*, Fuzzy Sets and Systems 518 (2025) 109485
(the PDF in this folder is `Fuzzy logic paper`).

## Layout

| Path | Contents |
|---|---|
| `fppa/fuzzy.py` | BMFs (6), (8); Yager t-conorm (1), (7); saturations (`exact`, paper's `tanh`, range-preserving `softmin`); analytic gradients |
| `fppa/environment.py` | Atomic obstacles (2)–(5), fuzzy path (8)–(12), environment (13)–(16) |
| `fppa/planner.py` | Contour level (17), speed bound (18), FPPA update (22), Algorithm 1 loop, APF baseline, unicycle signals (23) |
| `experiments/scenarios.py` | Paper cases, triangle variants, and destination-obstacle cases 4–5 |
| `experiments/make_all.py` | Refresh all static charts, GIFs, and results together |
| `experiments/make_figures.py` | Analysis PNG figures + `figures/results.json` |
| `experiments/make_paper_figures.py` | Paper-controller PNG figures using current scenarios + `figures/results_paper.json` |
| `experiments/make_animations.py` | GIFs, matching static path/signal charts, and `figures/animation_results.json` |
| `experiments/test_gradients.py` | Analytic vs. finite-difference gradient check |
| `slides/fppa_slides.tex` / `.pdf` | Beamer deck (compile from inside `slides/`) |

## Run

```bash
python3 -m experiments.test_gradients     # max rel. error ~2e-9
python3 -m experiments.test_animations    # sharp memberships, safe completion, crash counter and rendering
python3 -m experiments.make_all          # recommended after scenario edits: all charts, GIFs and results
python3 -m experiments.make_all --static-only  # refresh all static charts; keep existing GIFs
python3 -m experiments.make_animations   # six GIFs plus matching static path/signal charts
python3 -m experiments.make_animations --scenes case3_triangle_1 case3_triangle_2  # triangle pair only
python3 -m experiments.make_animations --scenes case4_stationary_destination case5_crossing_destination
python3 -m experiments.make_figures      # analysis static charts only
cd slides && pdflatex fppa_slides.tex && pdflatex fppa_slides.tex
```

Requires numpy, scipy, matplotlib (and Pillow for the GIF).

## Navigation animations

Run the commands from this directory. If dependencies are missing, install them
in a virtual environment with `python3 -m pip install numpy scipy matplotlib Pillow`.

| GIF in `figures/` | Scenario |
|---|---|
| `case2_animation.gif` | Curved Case 2 path with O7 starting at `(0, 10.2)` m |
| `case3_triangle_1.gif` | `triangle1()`; straight path with obstacle centers at `(50, −5)`, `(60, 5)`, `(70, −5)` m |
| `case3_triangle_2.gif` | `triangle2()`; straight path with obstacle centers at `(50, −10)`, `(60, 10)`, `(70, −10)` m |
| `case2_near_crisp_animation.gif` | Identical Case 2 geometry, fuzzy path and vehicle settings; only obstacle memberships sharpened |
| `case4_stationary_destination.gif` | Straight path; stationary O1 centered on the destination |
| `case5_crossing_destination.gif` | Straight path; perpendicular O1 timed to meet the nominal bot at the destination |

Each GIF scenario also writes `<scenario>_path.png` (trajectory and clearance)
and `<scenario>_signals.png` (speed, heading, acceleration, yaw rate) from the
same simulation log. The prefixes are `case2`, `case2_near_crisp`,
`case3_triangle_1`, `case3_triangle_2`, `case4_stationary_destination`, and
`case5_crossing_destination`. Static trajectory charts show the
full route with physical objects and membership at t = 0, moving-object tracks,
plain red object names, and any crash count. Motion charts also show crash counts
when applicable.

Use `make_all` after editing scenarios to refresh these charts together with
both original PNG sets (`f*.png`, `paper_*.png`) and their JSON results. The
original `f12_case3_*` / `paper_case3_*` charts still represent the paper's
12 m/s curved-path case; the two new triangle cases use their own prefixes.

Black circles show **physical radii** (`lam`), not the safety margin or
velocity-inflated fuzzy boundary. Plain red labels O1–O7 follow their objects,
without background boxes covering the field. The bot's
speed tag uses consecutive simulation positions divided by elapsed simulation
time, including both normal and tangential motion. Speed is unavailable at the
initial frame. The trail is the simulated trajectory; the dashed line is the
desired path. Each GIF includes the final sample. Playback is approximately 10×
simulation speed for the original cases (20 fps, every tenth 0.05 s sample) and
5× for the destination cases, to make the encounter easier to see.

After the first collision, a red **CRASHED · Crashes: N** message stays at the
top right. Each new contact with a physical object counts once; continuous
contact does not increment the counter every frame. Leaving and re-entering
counts again, and contacting two different objects counts separately. Counts
use every simulation sample, so collisions between displayed frames are still
included. Replaying or redrawing a frame does not change its count. Crash totals
and event times are also saved in `animation_results.json`.

O7 now starts 15 m from the graph's left edge at x = −15 m, so x = 0 m,
and travels right at `(3, 0)` m/s. Encounter timing, physical clearance and the
speed/position response relative to a run without O7 are recorded under
`case2.O7_encounter` in `animation_results.json` for the current saved geometry.

Case 2's O3 physical radius is the third entry (`radius[2]`) in
`CASE2_OBJECTS`. The separate 1.25 m safety margin is added to its physical
radius when forming the static fuzzy scale (e.g. 1 m gives a 2.25 m scale).
Both the fuzzy and obstacle-only near-crisp runs read this same radius array;
the triangle functions retain their own radius settings. Each results file
records the radii used for its run, so it can be checked after scenario edits.

The two fully fuzzy triangle scenes use the `triangle1()` and `triangle2()`
functions in `scenarios.py`, including their current `tanh` saturation defaults.
Both have a straight path from `(0, 0)` to `(120, 0)` m, radius-3 m objects and
1.25 m safety margins. The plots share axis limits so their spacing is easy to
compare. The passage report records the actual side of each obstacle. Selective
generation with `--scenes` updates these results while retaining results for the
other animations. The old single-triangle animation is no longer generated.

### Destination-obstacle cases

Both new cases use a straight desired path from `(0, 0)` to `(120, 0)` m,
nominal bot speed 5 m/s, and one radius-3 m obstacle with a 1.25 m safety margin.
They use the existing `tanh` fuzzy model and FPPA parameters.

- **Case 4:** O1 stays at `(120, 0)`, directly on the destination. A collision-free
  arrival is impossible while that point is occupied. The original controller
  follows the obstacle contour and can turn back; no waiting or goal-seeking
  controller has been added. The run ends at the 36 s horizon, marked
  `destination_occupied` rather than completed.
- **Case 5:** O1 starts at `(120, 120)` and moves at `(0, −5)` m/s. The nominal
  bot and O1 each start 120 m from the goal, with equal 5 m/s speeds and
  perpendicular routes. Without avoidance they coincide at `(120, 0)` at 24 s.
  A blue outline shows this nominal bot for reference; crash counts use only
  the actual FPPA bot. The actual bot slows and detours, then enters the clear
  goal region at about 25.45 s without a physical collision. Minimum clearance
  is about 0.59 m; the configured 1.25 m safety margin is not maintained throughout.

For these two endpoint tests only, `run_fppa(..., goal_mode="position",
goal_tol=2)` requires arrival within 2 m of the actual goal, with both the bot
and goal free of physical obstacles. This prevents passing alongside a blocked
destination from being reported as success. Other cases keep the original
projection-based stopping rule. The steering and speed equations are unchanged.
Static path/clearance and motion-signal charts are generated alongside both GIFs.

### Near-crisp comparison

The FPPA update equations, starting point, geometry, motion, timestep, fuzzy
path, all vehicle parameters, and `softmin` saturation are identical between
the two Case 2 animations. Only the obstacle bell exponent `b` changes:

| Parameter | Fuzzy | Near-crisp |
|---|---:|---:|
| Obstacle bell exponent `b` | 1.25 | 20 |
| Path membership exponent `k` | 1 | 1 |
| Contour exponent `d_o` | 1.25 | 1.25 |
| Speed exponent `d_v` | 0.5 | 0.5 |

Path atoms use `exp(-0.5 * s**k)`, with `s` the squared normalized distance.
`FuzzyPath(..., sharpness=1)` preserves the original Gaussian. Increasing `k`
sharpens the elliptical boundary. Stable power evaluation and analytic gradients
avoid overflow at obstacle/path centers and far into the tails.

Only the obstacle memberships are finite, smooth **near-crisp** approximations;
they are not binary sets. The Gaussian path and the original `d_o` / `d_v`
settings are restored. Completion, travel time, collisions and measured speeds
for the current geometry are recorded in `animation_results.json`. Steep
obstacle gradients can produce large normal corrections, because the original
FPPA speed bound limits the tangential term rather than the full displacement.
The controller is unchanged in both runs.
This demonstrates the effect of this chosen obstacle exponent, not a claim
that every crisp navigation method is inferior.

`figures/animation_results.json` records the full settings, completion, time,
clearances, measured speeds, acceleration statistics, triangle passages, and an
O7 comparison against a run without O7. `make_all` refreshes this file together
with `results.json` and `results_paper.json`, all PNG charts, and the GIFs.
Recompile the slide PDF separately after refreshing its PNG inputs.

## Assumptions (not stated in the paper)

- Case 1 path along x2 = 0 (as in Fig. 6; the text says x2 = 20).
- Case 2 global path digitised from Fig. 11; moving obstacles: O6 velocity (0, −3), O7 velocity (3, 0) for the updated animations (originally (2, 0)).
- Case 2/3 use b = 1.25 and γ = 0.75 from Case 1 (not listed for Case 2).
- Eq. (10) is implemented as H D Hᵀ (consistent with eq. (3)).
- The environment is evaluated as one m-ary union of all atoms, because the tanh-smoothed operator is not associative.
- Baseline is a generic path-guided APF, **not** the IAPF of the paper's ref. [37].
- Simulation stops when the projection on the path is within 1.5 segments of the end (level curves are closed).

## Main findings

- With the paper's `tanh` smoothing and γ = 0.75, the path core only reaches
  tanh(γ√(2π)) = 0.9545 (Poisson summation), so α = 0.99 is an infeasible set-point.
  The vehicle rides the ridge; in Case 3 it collides with O2 at t = 6.4 s.
- A feasible α (0.90), or the range-preserving `softmin` saturation with α = 0.99, completes all cases.
  Case 1: 78.1 s (paper ≈ 78 s). The predicted lateral offset is 1.1325 m; the simulated offset is 1.1341 m.
- The APF baseline stalls inside the concave obstacle at (96.1, 0).
