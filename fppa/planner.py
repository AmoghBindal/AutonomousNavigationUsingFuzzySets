"""FPPA update law (Sec. 3, Algorithm 1), a path-guided APF baseline, and the
unicycle signals of eq. (23) recovered by differentiating the planned path."""
import time
from dataclasses import dataclass

import numpy as np

from .fuzzy import R90


@dataclass
class FPPAParams:
    alpha: float = 0.99   # desired membership in the path without obstacles (eq. 17)
    d_o: float = 1.25     # contour-selection exponent (eq. 17)
    v_m: float = 5.0      # max speed on the path (eq. 18)
    d_v: float = 0.5      # speed-decay exponent (eq. 18)
    eta_n: float = 0.75   # normal (gradient) step size (eq. 22)
    t_s: float = 0.05     # sampling time; eta_t = v_max t_s
    sign: float = 1.0     # +1: tangent R(pi/2) grad/|grad|  (keeps obstacles on the vehicle's right)
    eps: float = 1e-9


def contour_level(mu_O, prm):
    """mu_th = alpha (1 - mu_O(x_k))^d_o   (eq. 17)."""
    return prm.alpha * (1.0 - mu_O) ** prm.d_o


def speed_bound(mu_O, prm):
    """v_max = v_m (1 - mu_O(x_k))^d_v   (eq. 18)."""
    return prm.v_m * (1.0 - mu_O) ** prm.d_v


def fppa_step(env, x, t, prm):
    """One planning interval:
        dx = -eta_n (mu_E - mu_th) grad mu_E  +  eta_t R(pi/2) grad mu_E / |grad mu_E|   (eq. 22)
    with mu_th and eta_t = v_max t_s held constant over the interval (zero-order hold)."""
    ev = env.evaluate(x, t)
    mu_th = contour_level(ev["mu_O"], prm)
    v_max = speed_bound(ev["mu_O"], prm)
    g = ev["g_E"]
    normal = -prm.eta_n * (ev["mu_E"] - mu_th) * g
    tangent = prm.sign * v_max * prm.t_s * (R90 @ g) / max(np.linalg.norm(g), prm.eps)
    info = dict(mu_E=ev["mu_E"], mu_O=ev["mu_O"], mu_th=mu_th, v_max=v_max,
                normal=normal, tangent=tangent)
    return normal + tangent, info


def goal_reached(env, x, t, goal_tol=None, goal_mode="progress"):
    """Legacy progress capture, or safe physical arrival for destination tests."""
    if goal_mode not in ("progress", "position"):
        raise ValueError("goal_mode must be 'progress' or 'position'")
    if goal_tol is None:
        goal_tol = 1.5 * (env.path.length - env.path.cum[-2])
    near = np.linalg.norm(x - env.path.w[-1]) < goal_tol
    if goal_mode == "position":
        if not near:
            return False
        if env.obs is None or len(env.obs) == 0:
            return True
        centers = env.obs.centers(t)
        return bool(np.all(np.linalg.norm(x - centers, axis=1) > env.obs.lam)
                    and np.all(np.linalg.norm(env.path.w[-1] - centers, axis=1) > env.obs.lam))
    return bool(env.path.progress(x)[0] >= env.path.length - goal_tol or near)


def run_fppa(env, x0, prm, t_final=200.0, goal_tol=None, *, goal_mode="progress"):
    """Simulate Algorithm 1 until the vehicle reaches the end of the path.

    The contours of mu_E are closed curves, so a contour follower that is not
    stopped wraps around the end of the path and drives back; we stop once the
    projection on the crisp path is within goal_tol (default: 1.5 segments) of the end.
    goal_mode='position' instead requires proximity to the actual goal and no
    physical obstacle contact. This changes termination only, not the update law.
    """
    if goal_mode not in ("progress", "position"):
        raise ValueError("goal_mode must be 'progress' or 'position'")
    x = np.asarray(x0, float).copy()
    L = env.path.length
    if goal_tol is None:
        goal_tol = 1.5 * (L - env.path.cum[-2])
    log = {k: [] for k in ("x", "t", "mu_E", "mu_O", "mu_th", "v_max", "normal", "tangent", "cpu")}
    t = 0.0
    while t <= t_final:
        c0 = time.perf_counter()
        dx, info = fppa_step(env, x, t, prm)
        cpu = time.perf_counter() - c0
        log["x"].append(x.copy()); log["t"].append(t); log["cpu"].append(cpu)
        for k in ("mu_E", "mu_O", "mu_th", "v_max", "normal", "tangent"):
            log[k].append(np.copy(info[k]))
        if goal_reached(env, x, t, goal_tol, goal_mode):
            break
        x = x + dx
        t += prm.t_s
    return {k: np.array(v) for k, v in log.items()}


@dataclass
class APFParams:
    """Path-guided APF (Khatib-type repulsion + attraction to a look-ahead point on
    the global path).  A generic baseline, *not* the exact IAPF of the paper's ref. [37]."""
    k_att: float = 1.0
    k_rep: float = 60.0
    rho0: float = 6.0       # influence distance of the repulsion (m)
    lookahead: float = 8.0  # carrot distance along the path (m)
    v_m: float = 5.0
    t_s: float = 0.05


def _carrot(path, x, lookahead):
    s, _ = path.progress(x)
    s = min(s + lookahead, path.length)
    i = min(np.searchsorted(path.cum, s) - 1, len(path.w) - 2)
    i = max(i, 0)
    u = (s - path.cum[i]) / (path.cum[i + 1] - path.cum[i])
    return path.w[i] + u * (path.w[i + 1] - path.w[i])


def run_apf(path, obstacles, x0, prm, t_final=200.0, goal_tol=1.0):
    x = np.asarray(x0, float).copy()
    xs, ts = [], []
    t = 0.0
    radius = obstacles.lam + obstacles.r
    while t <= t_final:
        xs.append(x.copy()); ts.append(t)
        if np.linalg.norm(x - path.w[-1]) < goal_tol:
            break
        F = prm.k_att * (_carrot(path, x, prm.lookahead) - x)
        d_vec = x - obstacles.centers(t)
        dist = np.maximum(np.linalg.norm(d_vec, axis=1) - radius, 1e-3)
        near = dist < prm.rho0
        if near.any():
            mag = prm.k_rep * (1 / dist[near] - 1 / prm.rho0) / dist[near] ** 2
            F = F + np.sum((mag / np.linalg.norm(d_vec[near], axis=1))[:, None] * d_vec[near], axis=0)
        speed = min(prm.v_m, np.linalg.norm(F))
        x = x + prm.t_s * speed * F / max(np.linalg.norm(F), 1e-12)
        t += prm.t_s
    return {"x": np.array(xs), "t": np.array(ts)}


def unicycle_signals(xs, t_s):
    """Recover v, psi, a, omega of the kinematic model (23) by finite differences."""
    d = np.diff(xs, axis=0)
    v = np.linalg.norm(d, axis=1) / t_s
    psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    a = np.diff(v) / t_s
    omega = np.diff(psi) / t_s
    return v, psi, a, omega


def min_clearance(xs, ts, obstacles, groups=None):
    """Distance from the vehicle to the physical (lam_i) boundary of every atom, over time.
    groups: optional list of index arrays to report per object."""
    d = np.stack([np.linalg.norm(x - obstacles.centers(t), axis=1) - obstacles.lam
                  for x, t in zip(xs, ts)])
    if groups is None:
        return d.min(axis=1)
    return np.stack([d[:, g].min(axis=1) for g in groups], axis=1)
