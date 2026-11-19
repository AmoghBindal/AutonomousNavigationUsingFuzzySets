"""Static path, clearance and motion charts from the same logs as the GIFs."""
from pathlib import Path

import numpy as np

from experiments.animations import annotate_crashes, crash_counts
from experiments.make_figures import env_background, save, signals_fig
from experiments.style import AQUA, BLUE, INK, INK2, ORANGE, RED, plt
from fppa import goal_reached, min_clearance


def write_static_charts(env, prm, groups, log, filename, title, xlim, ylim, *, goal_mode="progress", goal_tol=None):
    stem = Path(filename).stem.removesuffix("_animation")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axs[0]
    env_background(ax, env, xlim, ylim, n=500, cbar=False)
    ax.plot(*env.path.w.T, "--", color=INK2, lw=1, label="Desired path")
    ax.plot(*log["x"].T, color=AQUA, lw=2, label="Bot trajectory")
    ax.plot(*env.path.w[0], "s", color=INK, mfc="white", ms=6, label="Start")
    ax.plot(*env.path.w[-1], "*", color=ORANGE, ms=10, label="Destination")
    for i, (c, radius) in enumerate(zip(env.obs.c0, env.obs.lam)):
        ax.add_patch(plt.Circle(c, radius, fill=False, color=INK, lw=1.2))
        ax.annotate(f"O{i + 1}", (c[0], c[1] + radius), xytext=(0, 7), textcoords="offset points",
                    color=RED, fontsize=8, ha="center", va="bottom")
    # Moving-object tracks are clipped to the same spatial bounds as the GIF.
    end_centers = env.obs.centers(log["t"][-1])
    for c, end, vel in zip(env.obs.c0, end_centers, env.obs.vel):
        if np.linalg.norm(vel) > 0:
            ax.plot([c[0], end[0]], [c[1], end[1]], ":", color=RED, lw=0.8, alpha=0.65)
    clearance = min_clearance(log["x"], log["t"], env.obs, groups)
    contact = clearance.min(axis=1) <= 0
    if contact.any():
        ax.plot(*log["x"][contact].T, "x", color=RED, ms=4, label="Physical contact")
    annotate_crashes(ax, log, env.obs, groups)
    ax.set(xlim=xlim, ylim=ylim, xlabel="x (m)", ylabel="y (m)",
           title="Full trajectory; field and physical objects at t = 0")
    portrait = (ylim[1] - ylim[0]) > 0.7 * (xlim[1] - xlim[0])
    ax.legend(loc="upper left" if portrait else "lower right", fontsize=7)

    ax = axs[1]
    for i in range(clearance.shape[1]):
        ax.plot(log["t"], clearance[:, i], lw=1.1, label=f"O{i + 1}")
    ax.axhline(0, color=RED, lw=1, ls="--", label="Contact boundary")
    ax.set(xlabel="t (s)", ylabel="Physical clearance (m)", title="Clearance over the full run",
           ylim=(min(-0.5, float(clearance.min()) - 0.5), 15))
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=11)
    if goal_mode == "position":
        distance = np.linalg.norm(log["x"][-1] - env.path.w[-1])
        reached = goal_reached(env, log["x"][-1], log["t"][-1], goal_tol, goal_mode)
        fig.text(0.02, 0.015, f"Safe goal capture (within {goal_tol:g} m): {'yes' if reached else 'no'}"
                 f"; final goal distance: {distance:.2f} m.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04 if goal_mode == "position" else 0, 1, 1))
    save(fig, f"{stem}_path.png")
    signals_fig([("FPPA", log, BLUE)], f"{stem}_signals.png", f"{title}: motion signals", t_s=prm.t_s,
                crashes=int(crash_counts(log, env.obs, groups)[-1]))
