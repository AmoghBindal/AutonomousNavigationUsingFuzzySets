"""Case-study figures with the paper's controller settings (Sec. 4).

    python -m experiments.make_paper_figures      (from the project root)

All unions use Yager p = 1 with the tanh(.) saturation, alpha = 0.99, and the other
parameters listed in Sec. 4, with the current geometry/motion in scenarios.py.
No baseline is plotted (the paper's IAPF of ref. [37]
is not implemented).  Also redraws paper Fig. 1 (BMF).  Writes figures/paper_*.png and figures/results_paper.json.
"""
import json
import os

import numpy as np

from experiments.make_figures import OUT, env_background, field, grid, save, signals_fig, summarize
from experiments.animations import annotate_crashes, crash_counts
from experiments.scenarios import case1, case2
from experiments.style import AQUA, BLUE, INK, INK2, RED, plt
from fppa import bell_bmf, min_clearance, run_fppa

RESULTS = {}


def bmf_fig():
    """Paper Fig. 1: contour plots of the BMF (6) for b = 1 and b = 10."""
    X, Y, P = grid((-3, 3), (-3, 3), 300)
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, b in zip(axs, (1, 10)):
        cs = field(ax, X, Y, bell_bmf(P, b)[0], vmax=1, cbar=False)
        ax.set(title=f"$b={b}$", xlabel="$x_1$ (m)", ylabel="$x_2$ (m)")
    fig.colorbar(cs, ax=axs, fraction=0.046, pad=0.03, label="membership")
    save(fig, "paper_bmf.png")


def case1_figs():
    env, prm, groups = case1("tanh")
    log = run_fppa(env, env.path.w[0], prm, t_final=150)
    RESULTS["case1"] = summarize(env, prm, log, groups)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    env_background(ax, env, (-5, 210), (-42, 42), n=700)
    ax.plot(*env.path.w.T, ".", color=INK, ms=2, label="desired path (waypoints)")
    ax.plot(*log["x"].T, color=AQUA, lw=2, label="FPPA path")
    ax.plot(0, 0, "s", color=INK, mfc="none", ms=8, label="start")
    ax.plot(200, 0, "*", color=INK, ms=11, label="target")
    ax.legend(loc="lower right", fontsize=8)
    ax.set(xlabel="$x_1$ (m)", ylabel="$x_2$ (m)", title="Case study 1: FPPA path around a concave obstacle")
    save(fig, "paper_case1_path.png")
    signals_fig([("FPPA", log, BLUE)], "paper_case1_signals.png", "Case study 1: velocity, heading, acceleration, yaw rate")


def case23_figs():
    cpu_log = None
    for vm, tag in ((5.0, "case2"), (12.0, "case3")):
        env, prm, groups = case2(vm, "tanh")
        log = run_fppa(env, env.path.w[0], prm, t_final=200)
        res = summarize(env, prm, log, groups)
        prog = np.array([env.path.progress(x)[0] for x in log["x"]])
        # first pass: from the start until the vehicle first gets within 1 m of its furthest progress
        k1 = int(np.argmax(prog >= prog.max() - 1.0))
        first = {k: v[: k1 + 1] for k, v in log.items()}
        cl = min_clearance(first["x"], first["t"], env.obs, groups)
        hit = np.where(cl.min(axis=1) < 0)[0]
        res["first_pass"] = dict(
            t_end=float(first["t"][-1]), progress=float(prog[k1]),
            remaining_to_end=float(env.path.length - prog[k1]),
            min_clearance=[round(float(c), 2) for c in cl.min(axis=0)],
            objects_hit=[i + 1 for i in range(cl.shape[1]) if (cl[:, i] < 0).any()],
            first_collision_t=float(first["t"][hit[0]]) if len(hit) else None)
        RESULTS[tag] = res
        if vm == 5.0:
            cpu_log, n_atoms = log, len(env.obs) + len(env.path.p)

        name = "Case study 2" if vm == 5 else "Case study 3"
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[2.2, 1]))
        ax = axs[0]
        env_background(ax, env, (-15, 280), (-5, 120), n=700, cbar=False)
        ax.plot(*env.path.w.T, ".", color=INK, ms=3, label="desired path")
        ax.plot(*first["x"].T, color=AQUA, lw=1.6, label="FPPA path (first pass)")
        for i, (c, r) in enumerate(zip(env.obs.c0, env.obs.lam)):
            ax.add_patch(plt.Circle(c, r, fill=False, color=RED, lw=1.2))
            ax.annotate(f"$O_{i + 1}$", c + [r + 1, r], fontsize=8, color=RED)
        moving = np.linalg.norm(env.obs.vel, axis=1) > 0
        for c, v in zip(env.obs.c0[moving], env.obs.vel[moving]):
            ax.annotate("", c + 6 * v, c, arrowprops=dict(arrowstyle="->", color=RED))
        if len(hit):
            ax.plot(*first["x"][hit].T, "x", color=INK, ms=4, label="inside an object")
        ax.plot(*first["x"][-1], "o", color=INK, mfc="none", ms=8, label="furthest point, then turns back")
        ax.plot(*env.path.w[0], "s", color=INK, mfc="none", ms=7)
        ax.plot(*env.path.w[-1], "*", color=INK, ms=10)
        ax.legend(loc="lower right", fontsize=7.5)
        ax.set(xlabel="$x_1$ (m)", ylabel="$x_2$ (m)", title=f"{name}: $v_m={vm:g}$ m/s (obstacles at $t=0$)")
        annotate_crashes(ax, first, env.obs, groups)
        ax = axs[1]
        ax.plot(log["t"], prog, color=BLUE)
        ax.plot(first["t"][-1], prog[k1], "o", color=INK, mfc="none", ms=7)
        ax.axhline(env.path.length, color=INK2, ls="--", lw=1)
        ax.text(2, env.path.length - 25, "end of path", color=INK2, fontsize=8)
        ax.set(xlabel="$t$ (s)", ylabel="distance along path (m)", title="progress along the desired path (full run)")
        save(fig, f"paper_{tag}_path.png")
        signals_fig([("FPPA", first, BLUE)], f"paper_{tag}_signals.png",
                    f"{name}: velocity, heading, acceleration, yaw rate (first pass)",
                    crashes=int(crash_counts(first, env.obs, groups)[-1]))

    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.hist(1e3 * cpu_log["cpu"], bins=60, color=BLUE)
    ax.set(xlabel="planning time per step (ms)", ylabel="count",
           title=f"FPPA time per planning step (case 2, {n_atoms} atoms)")
    save(fig, "paper_cpu.png")


if __name__ == "__main__":
    bmf_fig()
    case1_figs()
    case23_figs()
    with open(os.path.join(OUT, "results_paper.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(json.dumps(RESULTS, indent=2))
