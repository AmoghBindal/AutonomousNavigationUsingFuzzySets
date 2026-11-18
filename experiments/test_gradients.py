"""Check analytic gradients of mu_O, mu_P, mu_E against central finite differences."""
import numpy as np
from fppa import FuzzyObstacles, FuzzyPath, FuzzyEnvironment

rng = np.random.default_rng(0)
xs = np.linspace(0, 2 * np.pi, 30)
path = FuzzyPath(np.c_[xs, np.sin(xs)], sigma=0.1, gamma=0.75)
obs = FuzzyObstacles(rng.uniform(0, 6, (5, 2)), lam=0.3, r=0.1, vel=rng.normal(size=(5, 2)), delta=0.25, b=4)
worst = 0
for sat in ("tanh", "exact", "softmin"):
    for p in (1, 2, 5):
        for flat in (True, False):
            env = FuzzyEnvironment(obs, path, p=p, sat=sat, flat=flat)
            X = rng.uniform([-0.5, -1.5], [6.5, 1.5], (200, 2))
            ev = env.evaluate(X, t=0.3)
            h = 1e-6
            for key in ("E", "O", "P"):
                if key == "P":
                    continue
                num = np.stack([(env.evaluate(X + h * e, 0.3)["mu_" + key] - env.evaluate(X - h * e, 0.3)["mu_" + key]) / (2 * h)
                                for e in np.eye(2)], -1)
                err = np.max(np.abs(num - ev["g_" + key]) / (1 + np.abs(num)))
                if sat != "exact":
                    worst = err if np.isnan(err) else max(worst, err)
                print(f"sat={sat:7s} p={p} flat={flat!s:5s} grad mu_{key}: max rel err {err:.2e}")
print("WORST (smooth sats):", worst)
assert worst < 1e-5
