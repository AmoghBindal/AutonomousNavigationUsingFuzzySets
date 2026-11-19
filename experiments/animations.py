"""Shared animation renderer; circles denote physical radii, not safety zones."""
from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from experiments.style import AQUA, BLUE, INK, INK2, MEMB, ORANGE, RED, plt
from fppa import goal_reached, min_clearance
from fppa.fuzzy import yager


OUT = Path(__file__).resolve().parents[1] / "figures"


def measured_speed(log):
    """Speed of the preceding simulation interval, aligned with each position.

    No preceding interval exists at t=0; the first frame therefore shows N/A.
    This is actual displacement speed, not the planner's tangential speed bound.
    """
    speed = np.full(len(log["t"]), np.nan)
    speed[1:] = np.linalg.norm(np.diff(log["x"], axis=0), axis=1) / np.diff(log["t"])
    return speed


def frame_indices(log, every):
    if every < 1:
        raise ValueError("every must be >= 1")
    return np.unique(np.r_[np.arange(0, len(log["t"]), every), len(log["t"]) - 1])


def crash_counts(log, obstacles, groups=None):
    """Cumulative entries into physical objects at every simulation sample.

    Sustained contact counts once per object. Leaving and re-entering counts
    again; simultaneous contacts with different objects count separately. Group
    atoms belonging to one physical object to avoid counting their overlap twice.
    Precomputing before frame subsampling also retains short collision episodes.
    """
    if groups is None:
        groups = [np.array([i]) for i in range(len(obstacles))]
    contact = min_clearance(log["x"], log["t"], obstacles, groups) <= 0
    entered = contact.copy()
    entered[1:] &= ~contact[:-1]
    return np.cumsum(entered.sum(axis=1))


def annotate_crashes(ax, log, obstacles, groups=None):
    """Add the final crash count to a static trajectory chart."""
    count = int(crash_counts(log, obstacles, groups)[-1])
    if count:
        ax.text(0.98, 0.98, f"CRASHED · Crashes: {count}", transform=ax.transAxes,
                ha="right", va="top", color=RED, fontsize=9, fontweight="bold", zorder=10)


class AnimationScene:
    """Persistent artists and cached static path values for inexpensive frames."""

    def __init__(self, env, log, *, groups=None, title="FPPA — Case 2", subtitle="",
                 xlim=(-15, 280), ylim=(-5, 120), resolution=600,
                 goal_mode="progress", goal_tol=None, nominal_speed=None, figsize=(12, 6.3)):
        self.env, self.log = env, log
        self.title = title
        self.speed = measured_speed(log)
        self.crash_counts = crash_counts(log, env.obs, groups)
        self.goal_mode, self.goal_tol = goal_mode, goal_tol
        self.nominal_speed = nominal_speed
        nx = resolution
        ny = max(2, round(nx * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])))
        X, Y = np.meshgrid(np.linspace(*xlim, nx), np.linspace(*ylim, ny))
        self.points = np.stack([X, Y], axis=-1)
        # Only values are needed for the background; cache the static path's
        # unsaturated union. Chunking avoids allocating a grid x atoms x 2 tensor.
        flat = self.points.reshape(-1, 2)
        path_norm = np.empty(len(flat))
        for start in range(0, len(flat), 4096):
            atoms, _ = env.path.atoms(flat[start:start + 4096])
            path_norm[start:start + len(atoms)] = (np.max(atoms, axis=-1) if np.isinf(env.p)
                                                  else np.sum(atoms ** env.p, axis=-1) ** (1 / env.p))
        self.path_norm = path_norm.reshape(X.shape)

        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=120)
        self.fig.subplots_adjust(left=0.06, right=0.96, bottom=0.25, top=0.87)
        ax = self.ax
        self.image = ax.imshow(self.membership(0), origin="lower", extent=(*xlim, *ylim),
                               cmap=MEMB, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        self.fig.colorbar(self.image, ax=ax, fraction=0.025, pad=0.025, label="Membership")
        ax.plot(*env.path.w.T, "--", color=INK2, lw=1, alpha=0.8, label="Desired path", zorder=3)
        self.trail, = ax.plot([], [], color=AQUA, lw=2, label="Bot trajectory", zorder=4)
        self.bot, = ax.plot([], [], "o", color=INK, ms=6, zorder=7)
        ax.plot(*env.path.w[0], "s", color=INK, mfc="white", ms=7, label="Start", zorder=5)
        ax.plot(*env.path.w[-1], "*", color=ORANGE, ms=12, label="Destination", zorder=5)
        self.nominal_bot = None
        if nominal_speed is not None:
            self.nominal_bot, = ax.plot([], [], "o", mfc="none", mec=BLUE, mew=1.5, ms=9,
                                        label="Nominal bot (no avoidance)", zorder=6)
        self.rings, self.labels = [], []
        centers = env.obs.centers(0)
        for i, (c, radius) in enumerate(zip(centers, env.obs.lam)):
            ring = plt.Circle(c, radius, fill=False, edgecolor=INK, lw=1.3, zorder=5)
            ax.add_patch(ring)
            label = ax.annotate(f"O{i + 1}", c, xytext=(0, 9), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, color=RED, zorder=6,
                                annotation_clip=True)
            self.rings.append(ring)
            self.labels.append(label)
        self.speed_tag = ax.annotate("", log["x"][0], xytext=(0, 20), textcoords="offset points",
                                     ha="center", va="bottom", fontsize=9, color=INK, zorder=9,
                                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=AQUA, alpha=0.95))
        self.status = self.fig.text(0.06, 0.065, "", fontsize=9, color=INK2)
        self.crash_banner = self.fig.text(0.95, 0.965, "", ha="right", va="top",
                                         fontsize=12, fontweight="bold", color=RED)
        self.fig.text(0.06, 0.025, subtitle or "Circles = physical objects; shading = fuzzy membership.",
                      fontsize=8, color=INK2)
        ax.set(xlim=xlim, ylim=ylim, xlabel="x (m)", ylabel="y (m)")
        ax.grid(False)
        handles, labels = ax.get_legend_handles_labels()
        self.fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.115),
                        ncol=4, fontsize=8)
        self.draw(0)

    def membership(self, t):
        atoms, _ = self.env.obs.atoms(self.points, t)
        obs_norm = (np.max(atoms, axis=-1) if np.isinf(self.env.p)
                    else np.sum(atoms ** self.env.p, axis=-1) ** (1 / self.env.p))
        if self.env.flat:
            if np.isinf(self.env.p):
                return np.maximum(self.path_norm, obs_norm)
            union = (self.path_norm ** self.env.p + obs_norm ** self.env.p) ** (1 / self.env.p)
            return self.env.sat(union)[0]
        return yager(np.stack([self.env.sat(self.path_norm)[0], self.env.sat(obs_norm)[0]], axis=-1),
                     self.env.p, sat=self.env.sat)

    def draw(self, k):
        t = self.log["t"][k]
        x = self.log["x"][k]
        self.image.set_data(self.membership(t))
        self.trail.set_data(*self.log["x"][:k + 1].T)
        self.bot.set_data([x[0]], [x[1]])
        if self.nominal_bot is not None:
            direction = self.env.path.w[-1] - self.env.path.w[0]
            distance = np.linalg.norm(direction)
            nominal = self.env.path.w[0] + direction * min(self.nominal_speed * t / distance, 1.0)
            self.nominal_bot.set_data([nominal[0]], [nominal[1]])
        centers = self.env.obs.centers(t)
        for c, radius, ring, label in zip(centers, self.env.obs.lam, self.rings, self.labels):
            ring.center = c
            # Labels sit just outside the physical rim, anchored in data space.
            label.xy = (c[0], c[1] + radius)
        self.speed_tag.xy = x
        value = f"{self.speed[k]:.2f} m/s" if k > 0 else "N/A (initial frame)"
        self.speed_tag.set_text(f"Bot · v = {value}")
        self.ax.set_title(f"{self.title}     t = {t:.2f} s", fontsize=11, pad=16)
        count = self.crash_counts[k]
        self.crash_banner.set_text(f"CRASHED · Crashes: {count}" if count else "")
        clearance = np.min(np.linalg.norm(x - centers, axis=1) - self.env.obs.lam)
        self.bot.set_color(RED if clearance <= 0 else INK)
        progress, _ = self.env.path.progress(x)
        status = (f"Progress: {progress:.1f} / {self.env.path.length:.1f} m"
                  f"     Physical clearance: {clearance:.2f} m"
                  + ("     COLLISION" if clearance <= 0 else ""))
        if self.goal_mode == "position":
            distance = np.linalg.norm(x - self.env.path.w[-1])
            occupied = np.any(np.linalg.norm(self.env.path.w[-1] - centers, axis=1) <= self.env.obs.lam)
            state = ("GOAL REACHED" if goal_reached(self.env, x, t, self.goal_tol, self.goal_mode)
                     else "DESTINATION OCCUPIED" if occupied else "Destination clear")
            status += f"\nGoal distance: {distance:.2f} m (tolerance {self.goal_tol:g} m)     {state}"
        self.status.set_text(status)


def animate(env, log, groups=None, name="case2_animation.gif", every=10, *,
            title="FPPA — Case 2 (fuzzy)", subtitle="", xlim=(-15, 280), ylim=(-5, 120),
            out_dir=OUT, goal_mode="progress", goal_tol=None, nominal_speed=None, figsize=(12, 6.3)):
    """Render every Nth step plus the final sample at 20 fps (10x by default)."""
    scene = AnimationScene(env, log, groups=groups, title=title, subtitle=subtitle, xlim=xlim, ylim=ylim,
                           goal_mode=goal_mode, goal_tol=goal_tol, nominal_speed=nominal_speed, figsize=figsize)
    output = Path(out_dir) / name
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = frame_indices(log, every)
    anim = FuncAnimation(scene.fig, scene.draw, frames=frames, cache_frame_data=False)
    anim.save(output, writer=PillowWriter(fps=20), dpi=120)
    plt.close(scene.fig)
    print("wrote", output, flush=True)
    return output
