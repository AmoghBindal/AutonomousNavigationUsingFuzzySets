"""Navigation GIFs, matching static charts and figures/animation_results.json.

    python3 -m experiments.make_animations [--scenes NAME ...] [--static-only]
"""
import argparse
import json
from dataclasses import asdict

import numpy as np

from experiments.animations import OUT, animate, crash_counts, measured_speed
from experiments.scenarios import case2, case4, case5, triangle1, triangle2
from experiments.static_charts import write_static_charts
from fppa import SATURATIONS, FuzzyObstacles, goal_reached, min_clearance, run_fppa


def run_summary(env, prm, groups, log, *, goal_mode="progress", goal_tol=None):
    speed = measured_speed(log)[1:]
    acceleration = np.diff(speed) / np.diff(log["t"])[1:]
    clearance = min_clearance(log["x"], log["t"], env.obs, groups)
    progress, _ = env.path.progress(log["x"][-1])
    if goal_tol is None:
        goal_tol = 1.5 * (env.path.length - env.path.cum[-2])
    completed = goal_reached(env, log["x"][-1], log["t"][-1], goal_tol, goal_mode)
    goal_clearance = np.min(np.linalg.norm(env.path.w[-1] - env.obs.centers(log["t"][-1]), axis=1) - env.obs.lam)
    stop_reason = ("goal_reached" if completed else "destination_occupied"
                   if goal_mode == "position" and goal_clearance <= 0 else "time_limit")
    counts = crash_counts(log, env.obs, groups)
    hits = np.flatnonzero(counts > 0)
    return dict(
        parameters=dict(planner=asdict(prm), obstacle_b=env.obs.b,
                        path_sharpness=env.path.sharpness,
                        goal_mode=goal_mode, goal_tolerance_m=float(goal_tol),
                        saturation=next((name for name, sat in SATURATIONS.items() if env.sat is sat), "custom"),
                        obstacle_centers_t0=env.obs.c0.tolist(),
                        physical_radii_m=env.obs.lam.tolist(), safety_margins_m=env.obs.r.tolist(),
                        obstacle_velocities_m_s=env.obs.vel.tolist(), velocity_inflation=env.obs.delta,
                        path_start=env.path.w[0].tolist(), path_destination=env.path.w[-1].tolist(),
                        path_sigma=1.5, path_gamma=0.75),
        completed=bool(completed), stop_reason=stop_reason,
        goal_distance_m=float(np.linalg.norm(log["x"][-1] - env.path.w[-1])),
        destination_occupied=bool(goal_clearance <= 0),
        final_time_s=float(log["t"][-1]), samples=len(log["t"]),
        final_position_m=log["x"][-1].tolist(), progress_m=float(progress),
        reference_path_length_m=float(env.path.length),
        travelled_distance_m=float(np.linalg.norm(np.diff(log["x"], axis=0), axis=1).sum()),
        min_clearance_m=float(clearance.min()),
        min_clearance_per_object_m={f"O{i + 1}": float(c) for i, c in enumerate(clearance.min(axis=0))},
        first_collision_time_s=float(log["t"][hits[0]]) if len(hits) else None,
        crash_count=int(counts[-1]),
        crash_events=[dict(time_s=float(log["t"][k]), new_crashes=int(n))
                      for k, n in enumerate(np.diff(counts, prepend=0)) if n],
        speed_m_s=dict(min=float(speed.min()), max=float(speed.max()), mean=float(speed.mean())),
        acceleration_m_s2=dict(abs_max=float(np.abs(acceleration).max()),
                              rms=float(np.sqrt(np.mean(acceleration ** 2)))),
    )


def o7_encounter(env, prm, log):
    """Measure the encounter and response against an otherwise identical run without O7."""
    distances = np.array([np.linalg.norm(x - env.obs.centers(t)[6]) - env.obs.lam[6]
                          for x, t in zip(log["x"], log["t"])])
    k = int(np.argmin(distances))
    baseline, _, _ = case2(sat="softmin")
    o = baseline.obs
    baseline.obs = FuzzyObstacles(o.c0[:6], o.lam[:6], r=o.r[:6], vel=o.vel[:6], delta=o.delta, b=o.b)
    without = run_fppa(baseline, baseline.path.w[0], prm, t_final=200)
    j = int(np.argmin(np.abs(without["t"] - log["t"][k])))
    return dict(time_s=float(log["t"][k]), physical_clearance_m=float(distances[k]),
                bot_position_m=log["x"][k].tolist(), obstacle_position_m=env.obs.centers(log["t"][k])[6].tolist(),
                obstacle_membership=float(env.obs.atoms(log["x"][k], log["t"][k])[0][6]),
                bot_speed_m_s=float(measured_speed(log)[k]),
                speed_without_O7_m_s=float(measured_speed(without)[j]),
                displacement_from_run_without_O7_m=float(np.linalg.norm(log["x"][k] - without["x"][j])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+",
                        choices=("case2", "case3_triangle_1", "case3_triangle_2", "case2_near_crisp",
                                 "case4_stationary_destination", "case5_crossing_destination"),
                        help="Render only these scenes, keeping existing results for the others.")
    parser.add_argument("--static-only", action="store_true",
                        help="Refresh matching static charts and results without re-encoding the GIFs.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "animation_results.json"
    results = json.loads(output.read_text()) if args.scenes and output.exists() else {}
    results["notes"] = [
        "Same FPPA update law for all runs; near-crisp uses finite smooth powers, not binary sets.",
        "Case 2 fuzzy/near-crisp use identical geometry, timing, path, vehicle settings and softmin saturation.",
        "Near-crisp changes only the obstacle bell exponent b from 1.25 to 20.",
        "Speed is measured from consecutive positions; initial speed is unavailable.",
        "Crashes count new physical contacts per object across all simulation samples; the banner persists after contact ends.",
        "20 fps; every tenth 0.05 s sample (10x playback), or every fifth for destination cases (5x), plus final sample.",
        "The two triangle variants use the tanh saturation.",
        "Destination cases use safe positional goal capture within 2 m and a 36 s maximum horizon; FPPA steering is unchanged.",
    ]
    scenes = [
        ("case2", case2(sat="softmin"), "case2_animation.gif", "FPPA — Case 2 (fuzzy)",
         "b = 1.25; path k = 1; d_o = 1.25; d_v = 0.5.  Circles = physical objects.",
         (-15, 280), (-5, 120)),
        ("case3_triangle_1", triangle1(), "case3_triangle_1.gif", "FPPA — Case 3: Triangle 1 (fuzzy)",
         "Straight desired path; Triangle 1.  Circles = physical objects; shading = membership.",
         (-5, 125), (-18, 24)),
        ("case3_triangle_2", triangle2(), "case3_triangle_2.gif", "FPPA — Case 3: Triangle 2 (fuzzy)",
         "Straight desired path; Triangle 2.  Circles = physical objects; shading = membership.",
         (-5, 125), (-18, 24)),
        ("case2_near_crisp", case2(sat="softmin", near_crisp=True), "case2_near_crisp_animation.gif",
         "FPPA — Case 2 (near-crisp obstacles only)",
         "Obstacle b = 20; fuzzy path k = 1; d_o = 1.25; d_v = 0.5.  Same FPPA and vehicle settings.",
         (-15, 280), (-5, 120)),
        ("case4_stationary_destination", case4(), "case4_stationary_destination.gif",
         "FPPA — Case 4: Stationary obstacle on the destination",
         "O1 occupies the goal. Physical radius = 3 m; the occupied destination cannot be reached safely.",
         (-10, 140), (-20, 30)),
        ("case5_crossing_destination", case5(), "case5_crossing_destination.gif",
         "FPPA — Case 5: Perpendicular encounter at the destination",
         "Both start 120 m from the goal at nominal 5 m/s. Unaltered routes meet at t = 24 s.",
         (-10, 140), (-25, 135)),
    ]
    for key, (env, prm, groups), filename, title, subtitle, xlim, ylim in scenes:
        if args.scenes and key not in args.scenes:
            continue
        destination_case = key in ("case4_stationary_destination", "case5_crossing_destination")
        goal_options = dict(goal_mode="position", goal_tol=2.0) if destination_case else {}
        log = run_fppa(env, env.path.w[0], prm, t_final=36 if destination_case else 200, **goal_options)
        if not all(np.isfinite(values).all() for values in log.values()):
            raise ValueError(f"non-finite simulation values in {key}")
        results[key] = run_summary(env, prm, groups, log, **goal_options)
        results[key]["gif"] = filename
        stride = 5 if destination_case else 10
        results[key]["playback"] = dict(fps=20, sample_stride=stride,
                                        simulation_speed_factor=20 * stride * prm.t_s)
        if key == "case2":
            results[key]["O7_encounter"] = o7_encounter(env, prm, log)
        if destination_case:
            arrival = env.path.length / prm.v_m
            results[key]["nominal_encounter"] = dict(
                bot_speed_m_s=prm.v_m, bot_distance_to_goal_m=env.path.length, arrival_time_s=arrival,
                obstacle_speed_m_s=float(np.linalg.norm(env.obs.vel[0])),
                obstacle_distance_to_goal_m=float(np.linalg.norm(env.obs.c0[0] - env.path.w[-1])),
                bot_position_at_arrival_m=env.path.w[-1].tolist(),
                obstacle_position_at_arrival_m=env.obs.centers(arrival)[0].tolist())
        if key.startswith("case3_triangle_"):
            results[key]["passages"] = []
            for i, c in enumerate(env.obs.c0):
                point = log["x"][np.argmin(abs(log["x"][:, 0] - c[0]))]
                results[key]["passages"].append(dict(
                    object=f"O{i + 1}", bot_position_m=point.tolist(),
                    side="above" if point[1] > c[1] else "below" if point[1] < c[1] else "level"))
        print(f"{key}: {results[key]['stop_reason']}, t={log['t'][-1]:.2f} s, "
              f"min clearance={results[key]['min_clearance_m']:.2f} m", flush=True)
        write_static_charts(env, prm, groups, log, filename, title, xlim, ylim, **goal_options)
        if not args.static_only:
            render_options = dict(every=stride)
            if key == "case5_crossing_destination":
                render_options.update(figsize=(9, 8.5), nominal_speed=prm.v_m)
            animate(env, log, groups, filename, title=title, subtitle=subtitle, xlim=xlim, ylim=ylim,
                    **goal_options, **render_options)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("wrote", output, flush=True)


if __name__ == "__main__":
    main()
