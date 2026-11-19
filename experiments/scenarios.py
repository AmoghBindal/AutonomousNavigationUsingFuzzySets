"""Scenario definitions taken from Sec. 4 of the paper (assumptions marked ASSUMED)."""
import numpy as np
from scipy.interpolate import CubicSpline

from fppa import FuzzyObstacles, FuzzyPath, FuzzyEnvironment, FPPAParams


def case1(sat="tanh"):
    # Straight path along x2 = 0 (Fig. 6; the text's "[0,20]^T" disagrees with the figure), M = 100.
    w = np.c_[np.linspace(0, 200, 100), np.zeros(100)]
    path = FuzzyPath(w, sigma=1.0, gamma=0.75)
    th = np.linspace(-3 * np.pi / 4, 3 * np.pi / 4, 97)          # concave arc, radius 30, centre (70, 0)
    arc = np.c_[70 + 30 * np.cos(th), 30 * np.sin(th)]
    obs = FuzzyObstacles(arc, lam=0.25, r=1.25, b=1.25, delta=0.1)
    prm = FPPAParams(alpha=0.99, d_o=1.25, v_m=5.0, d_v=0.5, eta_n=0.75, t_s=0.05)
    return FuzzyEnvironment(obs, path, p=1, sat=sat), prm, [np.arange(97)]


# ASSUMED: global path of Case Study 2 digitised from Fig. 11 and smoothed with a spline.
CASE2_KNOTS = np.array([[-5, 80], [20, 80], [45, 80], [62, 79.5], [71, 73], [75, 60], [76.5, 40],
                        [78, 22], [86, 12], [100, 10], [130, 10], [152, 12], [166, 22], [177, 40],
                        [187, 55], [198, 61], [210, 57], [224, 45], [238, 40], [251, 45],
                        [261, 60], [270, 82]], float)

# Table 1.  ASSUMED velocity directions (the paper gives speeds only): O6 crosses the
# path downward, O7 drives along the lower straight in the travel direction.
CASE2_OBJECTS = dict(
    # O7 starts 15 m from the plotted left edge (-15 m), allowing an encounter.
    pos=np.array([[69.5, 76.5], [69, 82.5], [186, 53], [192, 56], [185, 48], [45, 112], [0, 10.2]]),
    radius=np.array([2, 3, 3, 3, 3, 3, 3], float),
    vel=np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, -3], [3, 0]], float),
)


def sample_path(knots, spacing):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))])
    cs = CubicSpline(s, knots, bc_type="natural")
    dense = cs(np.linspace(0, s[-1], 4000))
    out = [dense[0]]
    for p in dense[1:]:
        if np.linalg.norm(p - out[-1]) >= spacing:
            out.append(p)
    if np.linalg.norm(dense[-1] - out[-1]) > 0.5 * spacing:
        out.append(dense[-1])
    return np.array(out)


def case2(v_m=5.0, sat="tanh", *, near_crisp=False):
    w = sample_path(CASE2_KNOTS, 5.0)
    # The comparison sharpens obstacles only; path and vehicle stay fuzzy.
    path = FuzzyPath(w, sigma=1.5, gamma=0.75)
    o = CASE2_OBJECTS
    obs = FuzzyObstacles(o["pos"], lam=o["radius"], r=1.25, vel=o["vel"], delta=0.1,
                         b=20 if near_crisp else 1.25)
    prm = FPPAParams(alpha=0.99, d_o=1.25, v_m=v_m, d_v=0.5, eta_n=0.75, t_s=0.05)
    return FuzzyEnvironment(obs, path, p=1, sat=sat), prm, [np.array([i]) for i in range(7)]


def triangle1(sat="tanh"):
    """Straight reference path with three static discs arranged for a weave."""
    w = np.c_[np.linspace(0, 120, 61), np.zeros(61)]
    path = FuzzyPath(w, sigma=1.5, gamma=0.75)
    obs = FuzzyObstacles([[50, -5], [60, 5], [70, -5]], lam=3, r=1.25, b=1.25)
    return FuzzyEnvironment(obs, path, p=1, sat=sat), FPPAParams(), [np.array([i]) for i in range(3)]

def triangle2(sat="tanh"):
    """Straight reference path with three static discs arranged for a weave."""
    w = np.c_[np.linspace(0, 120, 61), np.zeros(61)]
    path = FuzzyPath(w, sigma=1.5, gamma=0.75)
    obs = FuzzyObstacles([[50, -10], [60, 10], [70, -10]], lam=3, r=1.25, b=1.25)
    return FuzzyEnvironment(obs, path, p=1, sat=sat), FPPAParams(), [np.array([i]) for i in range(3)]


def case4(sat="tanh"):
    """Straight path with a stationary physical obstacle centered on the goal."""
    w = np.c_[np.linspace(0, 120, 61), np.zeros(61)]
    path = FuzzyPath(w, sigma=1.5, gamma=0.75)
    obs = FuzzyObstacles([[120, 0]], lam=3, r=1.25, b=1.25, delta=0.1)
    return FuzzyEnvironment(obs, path, p=1, sat=sat), FPPAParams(), [np.array([0])]


def case5(sat="tanh"):
    """Equal nominal travel time: bot rightward and O1 downward meet at t=24 s."""
    w = np.c_[np.linspace(0, 120, 61), np.zeros(61)]
    path = FuzzyPath(w, sigma=1.5, gamma=0.75)
    obs = FuzzyObstacles([[120, 120]], lam=3, r=1.25, b=1.25, vel=[[0, -5]], delta=0.1)
    return FuzzyEnvironment(obs, path, p=1, sat=sat), FPPAParams(v_m=5), [np.array([0])]
