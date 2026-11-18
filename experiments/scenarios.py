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
    pos=np.array([[69.5, 76.5], [69, 82.5], [186, 53], [192, 56], [185, 48], [45, 112], [79.5, 10.2]]),
    radius=np.array([2, 3, 2, 3, 3, 3, 3], float),
    vel=np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, -3], [2, 0]], float),
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


def case2(v_m=5.0, sat="tanh"):
    w = sample_path(CASE2_KNOTS, 5.0)
    path = FuzzyPath(w, sigma=1.5, gamma=0.75)
    o = CASE2_OBJECTS
    obs = FuzzyObstacles(o["pos"], lam=o["radius"], r=1.25, vel=o["vel"], delta=0.1, b=1.25)
    prm = FPPAParams(alpha=0.99, d_o=1.25, v_m=v_m, d_v=0.5, eta_n=0.75, t_s=0.05)
    return FuzzyEnvironment(obs, path, p=1, sat=sat), prm, [np.array([i]) for i in range(7)]
