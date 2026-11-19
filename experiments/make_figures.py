"""Reproduce the paper's figures and the additional mathematical figures.

    python -m experiments.make_figures          (from the project root)

All PNGs go to figures/, all numbers quoted in the slides go to figures/results.json.
"""
import json
import os

import numpy as np

from experiments.style import (AQUA, BLUE, GRID, INK, INK2, MEMB, ORANGE, RED, VIOLET, plt)
from experiments.scenarios import case1, case2
from experiments.animations import annotate_crashes
from fppa import (APFParams, FPPAParams, FuzzyEnvironment, FuzzyObstacles, FuzzyPath,
                  bell_bmf, min_clearance, run_apf, run_fppa, unicycle_signals, yager)
from fppa.fuzzy import make_sat_softmin, sat_tanh

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
RESULTS = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def grid(xlim, ylim, n=400):
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, int(n * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])) + 2)
    X, Y = np.meshgrid(x, y)
    return X, Y, np.stack([X, Y], -1)


def field(ax, X, Y, Z, levels=12, cbar=True, vmax=None):
    cs = ax.contourf(X, Y, Z, levels=np.linspace(0, vmax or max(Z.max(), 1e-9), levels + 1), cmap=MEMB)
    if cbar:
        plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.03, label=r"membership")
    ax.set_aspect("equal")
    ax.grid(False)
    return cs


# ---------------------------------------------------------------------------
# Section 2: fuzzy primitives
# ---------------------------------------------------------------------------
def fig_bell_bmf():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    r = np.linspace(0.01, 3, 600)
    for b, c in zip((1, 1.25, 4, 10), (BLUE, ORANGE, AQUA, VIOLET)):
        axs[0].plot(r, np.tanh(r ** (-2.0 * b)), color=c, label=f"$b={b}$")
    axs[0].axhline(np.tanh(1), color=INK2, lw=1, ls="--")
    axs[0].axvline(1, color=INK2, lw=1, ls="--")
    axs[0].text(1.05, np.tanh(1) + 0.03, r"$\alpha=\tanh 1\approx0.762$", color=INK2, fontsize=9)
    axs[0].set(xlabel=r"$\|\chi\|_2$", ylabel=r"$\mu(\chi)$", title=r"radial profile of $\mu=\tanh((\chi^T\chi)^{-b})$")
    axs[0].legend()
    X, Y, P = grid((-3, 3), (-3, 3), 300)
    for ax, b in zip(axs[1:], (1, 10)):
        field(ax, X, Y, bell_bmf(P, b)[0], vmax=1)
        ax.contour(X, Y, bell_bmf(P, b)[0], levels=[np.tanh(1)], colors=INK, linewidths=1)
        ax.set(title=f"$b={b}$ (black: $\\alpha$-cut at $\\tanh 1$ = unit circle)", xlabel="$\\chi_1$", ylabel="$\\chi_2$")
    save(fig, "f01_bell_bmf.png")


def fig_yager():
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
    a = np.linspace(0, 1, 400)
    b0 = 0.4
    for p, c in zip((1, 2, 5, np.inf), (BLUE, ORANGE, AQUA, VIOLET)):
        lab = r"$p=\infty$ (max)" if np.isinf(p) else (r"$p=1$ (bounded sum)" if p == 1 else f"$p={p}$")
        axs[0].plot(a, yager(np.stack([a, np.full_like(a, b0)], -1), p), color=c, label=lab)
    drastic = np.where(a == 0, b0, 1.0)
    axs[0].plot(a, drastic, color=INK2, lw=1, ls=":", label="drastic sum ($p\\to0^+$)")
    axs[0].plot(a, np.tanh(a + b0), color=RED, ls="--", label=r"$\tanh(a+b)$ (paper, $p=1$)")
    axs[0].set(xlabel="$a$", ylabel="$S_p(a,0.4)$", title="Yager t-conorm, second operand $b=0.4$", ylim=(0, 1.05))
    axs[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)

    # boundary law S(a,0)=a and the tanh defect
    axs[1].plot(a, a, color=BLUE, label=r"$S_p(a,0)=a$ (any $p$)")
    axs[1].plot(a, np.tanh(a), color=RED, ls="--", label=r"$\tanh(a)$")
    soft = make_sat_softmin(25)
    axs[1].plot(a, soft(a)[0], color=AQUA, ls="-.", label=r"softmin$_{\kappa=25}(a)$")
    axs[1].set(xlabel="$a$", ylabel="$S(a,0)$", title=r"neutral element 0: $\tanh$ breaks it")
    axs[1].legend(fontsize=8)

    # union of two fuzzy discs: conservativeness vs p
    X, Y, P = grid((-3, 3), (-2, 2), 300)
    o = FuzzyObstacles([[-0.9, 0], [0.9, 0]], lam=0.7, b=2)
    mu_i, _ = o.atoms(P)
    for p, c in zip((1, 2, np.inf), (BLUE, ORANGE, VIOLET)):
        axs[2].contour(X, Y, yager(mu_i, p), levels=[0.5], colors=c, linewidths=2)
        axs[2].plot([], [], color=c, label=("$p=\\infty$" if np.isinf(p) else f"$p={p}$"))
    axs[2].set_aspect("equal")
    axs[2].set(title=r"$0.5$-cut of $O_1\cup O_2$: smaller $p$ = larger obstacle", xlabel="$x_1$", ylabel="$x_2$")
    axs[2].legend(fontsize=8)
    save(fig, "f02_yager.png")


def fig_example1():
    c = np.array([[3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [5, 3], [5, 4], [5, 6], [5, 7]], float)
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.4))
    axs[0].add_patch(plt.Polygon([[2.5, 4], [4, 4], [4, 2], [6, 2], [6, 4], [7.5, 4], [7.5, 6], [6, 6], [6, 8],
                                  [4, 8], [4, 6], [2.5, 6]], fc=GRID, ec=INK2))
    axs[0].plot(*c.T, "o", color=BLUE, ms=6)
    for i, ci in enumerate(c):
        axs[0].annotate(f"$c_{i + 1}$", ci + [0.12, 0.12], fontsize=8, color=INK2)
    axs[0].set(xlim=(1.5, 8.5), ylim=(1.5, 8.5), title="object $O$ and atom centres", aspect="equal")
    X, Y, P = grid((0, 10), (0, 10), 300)
    for ax, v, dl, r, ttl in ((axs[1], [0, 0], 0, 0, "static: $v=0$"),
                               (axs[2], [2, 2], 0.25, 0.1, r"moving: $v=[2,2]^T$, $\delta=0.25$, $r=0.1$")):
        o = FuzzyObstacles(c, lam=0.75, r=r, vel=v, delta=dl, b=4)
        mu_i, _ = o.atoms(P)
        field(ax, X, Y, yager(mu_i, 1, sat=sat_tanh), vmax=1, cbar=ax is axs[2])
        ax.set(title=ttl, xlabel="$x_1$", ylabel="$x_2$")
    save(fig, "f03_example1_obstacle.png")


def example_path():
    xs = np.linspace(0, 2 * np.pi, 33)
    return FuzzyPath(np.c_[xs, np.sin(xs)], sigma=0.1, gamma=0.75)


def fig_example2():
    path = example_path()
    X, Y, P = grid((-0.5, 6.8), (-1.5, 1.5), 500)
    mu_j, _ = path.atoms(P)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5.4))
    field(axs[0], X, Y, mu_j.max(-1), vmax=1)
    for j in range(0, len(path.p), 4):
        axs[0].contour(X, Y, mu_j[..., j], levels=[0.6], colors=INK, linewidths=0.8)
    axs[0].plot(*path.w.T, ".", color=BLUE, ms=4)
    axs[0].set(title=r"individual fuzzy waypoints $\mu_{W_j}$ (black: 0.6-cuts of every 4th)", ylabel="$x_2$")
    field(axs[1], X, Y, yager(mu_j, 1, sat=sat_tanh), vmax=1)
    axs[1].set(title=r"fuzzy path $\mu_P=\tanh(\sum_j\mu_{W_j})$", xlabel="$x_1$", ylabel="$x_2$")
    save(fig, "f04_example2_path.png")


def example_env(sat="tanh"):
    obs = FuzzyObstacles([[np.pi, 0]], lam=0.3, r=0.05, b=4)
    return FuzzyEnvironment(obs, example_path(), p=1, sat=sat)


def fig_example3_and_rule():
    env = example_env()
    X, Y, P = grid((-0.5, 6.8), (-1.5, 1.5), 500)
    Z = env.evaluate(P)["mu_E"]
    prm = FPPAParams(alpha=0.92, d_o=1.25, v_m=0.5, d_v=0.5, eta_n=0.02, t_s=0.05)
    log = run_fppa(env, env.path.w[0], prm, t_final=60)
    RESULTS["example3"] = dict(T=float(log["t"][-1]), min_clearance=float(min_clearance(log["x"], log["t"], env.obs).min()))

    fig, ax = plt.subplots(figsize=(8, 3.4))
    field(ax, X, Y, Z, vmax=1)
    ax.contour(X, Y, Z, levels=[0.3, 0.5, 0.7, 0.92], colors=INK, linewidths=0.7)
    ax.add_patch(plt.Circle((np.pi, 0), 0.3, fill=False, color=INK, lw=1.2))
    ax.set(title=r"Example 3: $\mu_E=\mu_{\tilde O}\dot+\mu_P$ (contours 0.3, 0.5, 0.7, 0.92)", xlabel="$x_1$", ylabel="$x_2$")
    save(fig, "f05_example3_env.png")

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.contour(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.8, 0.92], cmap="viridis", linewidths=1)
    ax.add_patch(plt.Circle((np.pi, 0), 0.3, fill=False, color=INK, lw=1.2))
    x = log["x"]
    ax.plot(*x.T, color=INK2, lw=1)
    idx = np.arange(0, len(x), 6)
    nrm, tan = log["normal"][idx], log["tangent"][idx]
    ax.quiver(*x[idx].T, *(nrm * 40).T, color=BLUE, angles="xy", scale_units="xy", scale=1, width=0.003,
              label="normal term $\\times40$")
    ax.quiver(*x[idx].T, *tan.T, color=RED, angles="xy", scale_units="xy", scale=0.25, width=0.003,
              label="tangential term")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
    ax.set(title=r"update rule (22): normal and tangential components (cf. paper Fig. 5)", xlabel="$x_1$", ylabel="$x_2$")
    save(fig, "f06_update_rule.png")

    # Lyapunov check: V = 1/2 (mu_E - mu_th)^2 and orthogonality of the tangent term
    V = 0.5 * (log["mu_E"] - log["mu_th"]) ** 2
    g = np.stack([env.evaluate(xi)["g_E"] for xi in x])
    ortho = np.abs(np.sum(log["tangent"] * g, 1)) / (np.linalg.norm(log["tangent"], axis=1) * np.linalg.norm(g, axis=1))
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3))
    axs[0].semilogy(log["t"], V + 1e-16, color=BLUE)
    axs[0].set(xlabel="$t$ (s)", ylabel=r"$V_k=\frac{1}{2}(\mu_E(x_k)-\mu_{th})^2$", title="Lyapunov function along the run")
    axs[1].plot(log["t"], log["mu_E"], color=BLUE, label=r"$\mu_E(x_k)$")
    axs[1].plot(log["t"], log["mu_th"], color=ORANGE, ls="--", label=r"$\mu_{th}=\alpha(1-\mu_{\tilde O})^{d_o}$")
    axs[1].plot(log["t"], log["mu_O"], color=AQUA, label=r"$\mu_{\tilde O}(x_k)$")
    axs[1].set(xlabel="$t$ (s)", title="contour tracking")
    axs[1].legend(fontsize=8)
    axs[2].semilogy(log["t"], ortho + 1e-17, color=VIOLET)
    axs[2].set(xlabel="$t$ (s)", ylabel=r"$|\cos\angle(t_k,\nabla\mu_E)|$", title="tangent term $\\perp$ gradient")
    save(fig, "f07_lyapunov.png")
    RESULTS["example3"]["max_cos_tangent_grad"] = float(ortho.max())


def fig_tuning():
    m = np.linspace(0, 1, 400)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for d, c in zip((0.5, 0.9978, 1.25, 2.0), (BLUE, ORANGE, AQUA, VIOLET)):
        axs[0].plot(m, 0.99 * (1 - m) ** d, color=c, label=f"$d_o={d}$")
    axs[0].plot([0.99], [0.01], "o", color=INK, ms=6)
    axs[0].annotate(r"$(0.99,\,0.01)\Rightarrow d_o=\frac{\ln(0.01/0.99)}{\ln 0.01}=0.9978$", (0.99, 0.01),
                    (0.08, 0.15), fontsize=8.5, arrowprops=dict(arrowstyle="->", color=INK2))
    axs[0].set(xlabel=r"$\mu_{\tilde O}(x_k)$", ylabel=r"$\mu_{th}$", title=r"contour selection (17), $\alpha=0.99$")
    axs[0].legend(fontsize=8)
    for d, c in zip((0.25, 0.5, 1.0, 2.0), (BLUE, ORANGE, AQUA, VIOLET)):
        axs[1].plot(m, 10 * (1 - m) ** d, color=c, label=f"$d_v={d}$")
    axs[1].plot([0.99], [1.0], "o", color=INK, ms=6)
    axs[1].annotate(r"$v_m=10,\ v_{max}=1 \Rightarrow d_v=\frac{\ln 0.1}{\ln 0.01}=0.5$", (0.99, 1.0), (0.05, 2.0),
                    fontsize=8.5, arrowprops=dict(arrowstyle="->", color=INK2))
    axs[1].set(xlabel=r"$\mu_{\tilde O}(x_k)$", ylabel=r"$v_{max}$ (m/s)", title="speed bound (18), $v_m=10$")
    axs[1].legend(fontsize=8)
    save(fig, "f08_tuning.png")


# ---------------------------------------------------------------------------
# Our analysis: the level of the path core (Poisson summation)
# ---------------------------------------------------------------------------
def fig_core_level():
    gam = np.linspace(0.3, 1.6, 300)
    k = np.arange(-60, 61)
    exact_mid = np.array([np.exp(-k ** 2 / (2 * g ** 2)).sum() for g in gam])
    exact_wp = np.array([np.exp(-(k + 0.5) ** 2 / (2 * g ** 2)).sum() for g in gam])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4))
    axs[0].plot(gam, gam * np.sqrt(2 * np.pi), color=BLUE, label=r"$\gamma\sqrt{2\pi}$ (Poisson, leading term)")
    axs[0].plot(gam, exact_mid, color=ORANGE, ls="--", label=r"$\sum_k e^{-k^2/2\gamma^2}$ (at $p_j$)")
    axs[0].plot(gam, exact_wp, color=AQUA, ls=":", label=r"$\sum_k e^{-(k+\frac{1}{2})^2/2\gamma^2}$ (at $w_j$)")
    axs[0].axvline(0.75, color=INK2, lw=1)
    axs[0].set(xlabel=r"overlap $\gamma$", ylabel=r"$\Sigma(\gamma)$", title=r"core value of $\sum_j\mu_{W_j}$ on a straight path")
    axs[0].legend(fontsize=8)
    axs[1].plot(gam, np.tanh(gam * np.sqrt(2 * np.pi)), color=RED, label=r"$\sup\mu_P=\tanh(\gamma\sqrt{2\pi})$")
    axs[1].plot(gam, np.minimum(1, gam * np.sqrt(2 * np.pi)), color=BLUE, label=r"exact sat: $\min(1,\gamma\sqrt{2\pi})$")
    axs[1].axhline(0.99, color=INK2, ls="--", lw=1)
    g_star = np.arctanh(0.99) / np.sqrt(2 * np.pi)
    axs[1].axvline(g_star, color=RED, lw=1, ls=":")
    axs[1].text(g_star + 0.02, 0.62, f"$\\gamma^*={g_star:.3f}$", color=RED, fontsize=9)
    axs[1].plot([0.75], [np.tanh(0.75 * np.sqrt(2 * np.pi))], "o", color=INK)
    axs[1].annotate(r"paper: $\gamma=0.75\Rightarrow 0.955<\alpha=0.99$", (0.75, 0.955), (0.32, 0.75), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color=INK2))
    axs[1].set(xlabel=r"overlap $\gamma$", ylabel="level", title=r"feasibility of $\mu_{th}=\alpha$", ylim=(0.5, 1.02))
    axs[1].legend(fontsize=8, loc="lower right")
    save(fig, "f09_core_level.png")
    ripple = 2 * np.exp(-2 * np.pi ** 2 * 0.75 ** 2)
    RESULTS["core"] = dict(sum_mid=float(np.exp(-k ** 2 / (2 * 0.75 ** 2)).sum()), poisson=0.75 * np.sqrt(2 * np.pi),
                           ripple_rel=float(ripple), tanh_core=float(np.tanh(0.75 * np.sqrt(2 * np.pi))),
                           gamma_star=float(g_star))


# ---------------------------------------------------------------------------
# Case studies
# ---------------------------------------------------------------------------
def signals_fig(runs, name, title, t_s=0.05, *, crashes=0):
    fig, axs = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True)
    for (lab, log, c) in runs:
        v, psi, a, om = unicycle_signals(log["x"], t_s)
        t = log["t"]
        axs[0, 0].plot(t[1:], v, color=c, lw=1.5, label=lab)
        axs[0, 1].plot(t[1:], psi, color=c, lw=1.5, label=lab)
        axs[1, 0].plot(t[2:], a, color=c, lw=1.2, label=lab)
        axs[1, 1].plot(t[2:], om, color=c, lw=1.2, label=lab)
    for ax, yl in zip(axs.ravel(), ("$v$ (m/s)", r"$\psi$ (rad)", "$a$ (m/s$^2$)", r"$\omega$ (rad/s)")):
        ax.set_ylabel(yl)
    for ax in axs[1]:
        ax.set_xlabel("$t$ (s)")
    if len(runs) > 1:
        axs[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    if crashes:
        fig.subplots_adjust(top=0.81)
        fig.text(0.98, 0.90, f"CRASHED · Crashes: {crashes}", ha="right", va="top",
                 fontsize=9, fontweight="bold", color=RED)
    save(fig, name)


def env_background(ax, env, xlim, ylim, t=0.0, n=500, cbar=True):
    X, Y, P = grid(xlim, ylim, n)
    # Keep large grids bounded in memory even for the 196-atom concave scene.
    points = P.reshape(-1, 2)
    values = np.empty(len(points))
    for start in range(0, len(points), 4096):
        batch = points[start:start + 4096]
        values[start:start + len(batch)] = env.evaluate(batch, t)["mu_E"]
    field(ax, X, Y, values.reshape(X.shape), vmax=1, cbar=cbar)


def summarize(env, prm, log, groups):
    v, psi, a, om = unicycle_signals(log["x"], prm.t_s)
    s, _ = env.path.progress(log["x"][-1])
    cl = min_clearance(log["x"], log["t"], env.obs, groups).min(0)
    return dict(T=float(log["t"][-1]), progress=float(s), length=float(env.path.length),
                completed=bool(s >= env.path.length - 1.5 * (env.path.length - env.path.cum[-2])),
                min_clearance=[round(float(c), 2) for c in cl], v_min=float(v.min()), v_max=float(v.max()),
                a_absmax=float(np.abs(a).max()), omega_absmax=float(np.abs(om).max()),
                cpu_ms_median=float(1e3 * np.median(log["cpu"])), cpu_ms_p99=float(1e3 * np.percentile(log["cpu"], 99)),
                physical_radii_m=env.obs.lam.tolist(), obstacle_centers_t0=env.obs.c0.tolist(),
                obstacle_velocities_m_s=env.obs.vel.tolist())


def fig_case1():
    out = {}
    env, prm, groups = case1("softmin")
    log = run_fppa(env, env.path.w[0], prm, t_final=150)
    out["softmin"] = summarize(env, prm, log, groups)
    env_t, prm_t, _ = case1("tanh")
    log_t = run_fppa(env_t, env_t.path.w[0], prm_t, t_final=150)
    out["tanh"] = summarize(env_t, prm_t, log_t, groups)
    apf = run_apf(env.path, env.obs, env.path.w[0], APFParams(), t_final=150)
    out["apf_end"] = apf["x"][-1].round(2).tolist()
    straight = (log["x"][:, 0] > 10) & (log["x"][:, 0] < 40)
    out["offset_softmin_sim"] = float(log["x"][straight, 1].mean())
    out["offset_softmin_theory"] = float(np.sqrt(2 * np.log(0.75 * np.sqrt(2 * np.pi) / 0.99)))
    RESULTS["case1"] = out

    fig, ax = plt.subplots(figsize=(10, 4.2))
    env_background(ax, env, (-5, 210), (-42, 42), n=700)
    ax.plot(*env.path.w.T, ".", color=INK, ms=2, label="desired path (waypoints)")
    ax.plot(*log["x"].T, color=AQUA, lw=2, label="FPPA")
    ax.plot(*apf["x"].T, color=ORANGE, lw=2, ls="--", label="path-guided APF (stalls)")
    ax.plot(*apf["x"][-1], "x", color=ORANGE, ms=10, mew=2)
    ax.plot(0, 0, "s", color=INK, mfc="none", ms=8)
    ax.plot(200, 0, "*", color=INK, ms=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.set(xlabel="$x_1$ (m)", ylabel="$x_2$ (m)", title="Case study 1: concave obstacle (local-minimum trap for APF)")
    save(fig, "f10_case1_path.png")
    signals_fig([("FPPA", log, BLUE)], "f11_case1_signals.png", "Case study 1: unicycle signals of the FPPA path")
    return log


def fig_case23():
    prev_tanh = None
    for vm, tag in ((5.0, "case2"), (12.0, "case3")):
        res = {}
        env, prm, groups = case2(vm, "softmin")
        log = run_fppa(env, env.path.w[0], prm, t_final=200)
        res["softmin"] = summarize(env, prm, log, groups)
        env_t, prm_t, _ = case2(vm, "tanh")
        log_t = run_fppa(env_t, env_t.path.w[0], prm_t, t_final=200)
        res["tanh_alpha0.99"] = summarize(env_t, prm_t, log_t, groups)
        prm_f = FPPAParams(**{**prm_t.__dict__, "alpha": 0.90})
        log_f = run_fppa(env_t, env_t.path.w[0], prm_f, t_final=200)
        res["tanh_alpha0.90"] = summarize(env_t, prm_f, log_f, groups)
        RESULTS[tag] = res
        if vm == 5.0:
            prev_tanh = (env_t, log_t)

        fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[2.2, 1]))
        ax = axs[0]
        env_background(ax, env, (-15, 280), (-5, 120), n=700, cbar=False)
        ax.plot(*env.path.w.T, ".", color=INK, ms=3, label="desired path")
        ax.plot(*log["x"].T, color=AQUA, lw=2, label=r"FPPA (softmin, $\alpha=0.99$)")
        ax.plot(*log_f["x"].T, color=BLUE, lw=1.3, ls="--", label=r"FPPA ($\tanh$, $\alpha=0.90$)")
        for i, (c, r) in enumerate(zip(env.obs.c0, env.obs.lam)):
            ax.add_patch(plt.Circle(c, r, fill=False, color=RED, lw=1.2))
            ax.annotate(f"$O_{i + 1}$", c + [r + 1, r], fontsize=8, color=RED)
        moving = np.linalg.norm(env.obs.vel, axis=1) > 0
        for c, v in zip(env.obs.c0[moving], env.obs.vel[moving]):
            ax.annotate("", c + 6 * v, c, arrowprops=dict(arrowstyle="->", color=RED))
        ax.legend(loc="lower right", fontsize=7.5)
        ax.set(xlabel="$x_1$ (m)", ylabel="$x_2$ (m)", title=f"{'Case 2' if vm == 5 else 'Case 3'}: $v_m={vm:g}$ m/s (field at $t=0$)")
        annotate_crashes(ax, log, env.obs, groups)
        ax = axs[1]
        cl = min_clearance(log["x"], log["t"], env.obs, groups)
        for i, c in enumerate((BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", VIOLET)):
            ax.plot(log["t"], cl[:, i], color=c, lw=1.3, label=f"$O_{i + 1}$")
        ax.axhline(1.25, color=INK2, ls="--", lw=1)
        ax.text(1, 1.6, "$r_i=1.25$ m", color=INK2, fontsize=8)
        ax.set(ylim=(0, 15), xlabel="$t$ (s)", ylabel="distance to object (m)", title="clearance (softmin run)")
        ax.legend(fontsize=7, ncol=2)
        save(fig, f"f12_{tag}_path.png")
        signals_fig([(r"softmin $\alpha=0.99$", log, AQUA), (r"$\tanh$ $\alpha=0.90$", log_f, BLUE)],
                    f"f13_{tag}_signals.png", f"{'Case 2' if vm == 5 else 'Case 3'}: unicycle signals")

        if vm == 12.0:
            fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
            env_background(axs[0], env_t, (-15, 280), (-5, 120), n=600, cbar=False)
            axs[0].plot(*log_t["x"].T, color=RED, lw=1.3, label=r"paper-faithful: $\tanh$, $\alpha=0.99$")
            for c, r in zip(env.obs.c0, env.obs.lam):
                axs[0].add_patch(plt.Circle(c, r, fill=False, color=INK, lw=1))
            cl_t = min_clearance(log_t["x"], log_t["t"], env.obs)
            hit = np.where(cl_t < 0)[0]
            if len(hit):
                axs[0].plot(*log_t["x"][hit].T, "x", color=INK, ms=4, label="inside an object")
                RESULTS[tag]["tanh_alpha0.99"]["first_collision_t"] = float(log_t["t"][hit[0]])
            axs[0].legend(loc="lower right", fontsize=8)
            axs[0].set(title=r"case 3, $\tanh$, $\alpha=0.99>\sup\mu_E$: rides the ridge, collides, reverses",
                       xlabel="$x_1$ (m)", ylabel="$x_2$ (m)")
            annotate_crashes(axs[0], log_t, env_t.obs, groups)
            prog = np.array([env_t.path.progress(x)[0] for x in log_t["x"]])
            prog_f = np.array([env_t.path.progress(x)[0] for x in log_f["x"]])
            prog2 = np.array([env_t.path.progress(x)[0] for x in prev_tanh[1]["x"]])
            axs[1].plot(log_t["t"], prog, color=RED, label=r"case 3: $\tanh$, $\alpha=0.99$")
            axs[1].plot(prev_tanh[1]["t"], prog2, color=ORANGE, ls=":", label=r"case 2: $\tanh$, $\alpha=0.99$")
            axs[1].plot(log_f["t"], prog_f, color=BLUE, ls="--", label=r"case 3: $\tanh$, $\alpha=0.90$")
            axs[1].axhline(env_t.path.length, color=INK2, lw=1)
            axs[1].set(xlabel="$t$ (s)", ylabel="arc-length progress $s(x_k)$ (m)", title="progress along path")
            axs[1].legend(fontsize=8)
            save(fig, "f14_tanh_failure.png")

        if vm == 5.0:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.hist(1e3 * log["cpu"], bins=60, color=BLUE)
            ax.set(xlabel="planning time per step (ms)", ylabel="count",
                   title=f"FPPA step cost (Python/NumPy, {len(env.obs) + len(env.path.p)} atoms)")
            save(fig, "f15_cpu.png")


if __name__ == "__main__":
    fig_bell_bmf()
    fig_yager()
    fig_example1()
    fig_example2()
    fig_example3_and_rule()
    fig_tuning()
    fig_core_level()
    fig_case1()
    fig_case23()
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(json.dumps(RESULTS, indent=2))
