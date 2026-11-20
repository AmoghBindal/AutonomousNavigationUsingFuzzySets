"""Analytic gradients of mu_O and mu_E against central finite differences."""
import itertools
import unittest

import numpy as np

from fppa import FuzzyEnvironment, FuzzyObstacles, FuzzyPath


class GradientTests(unittest.TestCase):
    def test_environment_gradients_match_finite_differences(self):
        rng = np.random.default_rng(0)
        xs = np.linspace(0, 2 * np.pi, 30)
        path = FuzzyPath(np.c_[xs, np.sin(xs)], sigma=0.1, gamma=0.75)
        obs = FuzzyObstacles(rng.uniform(0, 6, (5, 2)), lam=0.3, r=0.1, vel=rng.normal(size=(5, 2)),
                             delta=0.25, b=4)
        h, t = 1e-6, 0.3
        # The exact saturation min(1, .) is not differentiable at its kink, so only smooth ones are checked.
        for sat, p, flat in itertools.product(("tanh", "softmin"), (1, 2, 5), (True, False)):
            env = FuzzyEnvironment(obs, path, p=p, sat=sat, flat=flat)
            X = rng.uniform([-0.5, -1.5], [6.5, 1.5], (200, 2))
            ev = env.evaluate(X, t)
            for key in ("E", "O"):
                with self.subTest(sat=sat, p=p, flat=flat, mu=key):
                    num = np.stack([(env.evaluate(X + h * e, t)["mu_" + key]
                                     - env.evaluate(X - h * e, t)["mu_" + key]) / (2 * h)
                                    for e in np.eye(2)], -1)
                    err = np.max(np.abs(num - ev["g_" + key]) / (1 + np.abs(num)))
                    self.assertLess(err, 1e-5)


if __name__ == "__main__":
    unittest.main()
