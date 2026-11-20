"""Membership, simulation, crash-counter and renderer checks."""
import unittest

import numpy as np

from experiments.animations import AnimationScene, RED, crash_counts, frame_indices, measured_speed, plt
from experiments.scenarios import case2, case4, case5, triangle1, triangle2
from fppa import FuzzyObstacles, goal_reached, run_fppa, min_clearance
from fppa.fuzzy import bell_bmf, gauss_bmf


class MembershipTests(unittest.TestCase):
    def test_steep_analytic_gradients(self):
        # Probe the sharp transition itself, rather than only the flat tails.
        r = np.linspace(0.8, 1.2, 100)
        chi = np.c_[r * np.cos(0.7), r * np.sin(0.7)]
        for bmf in (lambda x: bell_bmf(x, 20), lambda x: gauss_bmf(x, 10)):
            _, grad = bmf(chi)
            h = 1e-6
            numerical = np.stack([(bmf(chi + h * e)[0] - bmf(chi - h * e)[0]) / (2 * h)
                                  for e in np.eye(2)], axis=-1)
            np.testing.assert_allclose(grad, numerical, atol=2e-7, rtol=2e-6)

    def test_extreme_points_and_powers_remain_finite(self):
        chi = np.array([[0., 0.], [1e-15, 0.], [1., 0.], [1e3, 0.], [1e100, 0.]])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for b in (1.25, 20, 1000):
                for bmf in (bell_bmf, gauss_bmf):
                    mu, grad = bmf(chi, b)
                    self.assertTrue(np.isfinite(mu).all() and np.isfinite(grad).all())
                    self.assertTrue(((mu >= 0) & (mu <= 1)).all())
                    np.testing.assert_array_equal(grad[0], [0, 0])

    def test_default_memberships_preserved(self):
        chi = np.random.default_rng(8).normal(size=(200, 2))
        s = np.maximum(np.sum(chi ** 2, axis=-1), 1e-9)
        for b in (1.25, 4):
            expected = np.tanh(s ** (-b))
            expected_grad = ((1 - expected ** 2) * (-2 * b) * s ** (-b - 1))[:, None] * chi
            mu, grad = bell_bmf(chi, b)
            np.testing.assert_allclose(mu, expected, atol=1e-15)
            np.testing.assert_allclose(grad, expected_grad, atol=2e-14)
        mu, grad = gauss_bmf(chi)
        expected = np.exp(-0.5 * np.sum(chi ** 2, axis=-1))
        np.testing.assert_array_equal(mu, expected)
        np.testing.assert_array_equal(grad, -expected[:, None] * chi)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {}
        for name, builder in (("case2", lambda: case2(sat="softmin")),
                              ("triangle1", triangle1), ("triangle2", triangle2),
                              ("near_crisp", lambda: case2(sat="softmin", near_crisp=True)),
                              ("case4", case4), ("case5", case5)):
            env, prm, groups = builder()
            options = dict(t_final=36, goal_mode="position", goal_tol=2) if name in ("case4", "case5") else dict(t_final=200)
            log = run_fppa(env, env.path.w[0], prm, **options)
            cls.runs[name] = env, prm, groups, log

    def test_fuzzy_scenes_finish_safely(self):
        for name in ("case2", "triangle1", "triangle2"):
            env, prm, groups, log = self.runs[name]
            progress, _ = env.path.progress(log["x"][-1])
            self.assertGreaterEqual(progress, env.path.length - 1.5 * (env.path.length - env.path.cum[-2]))
            self.assertGreater(min_clearance(log["x"], log["t"], env.obs).min(), 0)

    def test_comparison_geometry_and_numerical_stability(self):
        env, prm, _, _ = self.runs["case2"]
        sharp, sharp_prm, _, log = self.runs["near_crisp"]
        np.testing.assert_array_equal(env.path.w, sharp.path.w)
        for field in ("c0", "lam", "r", "vel"):
            np.testing.assert_array_equal(getattr(env.obs, field), getattr(sharp.obs, field))
        np.testing.assert_array_equal(env.obs.c0[6], [0, 10.2])
        np.testing.assert_array_equal(env.obs.vel[6], [3, 0])
        self.assertEqual(prm, sharp_prm)
        self.assertEqual(env.path.sharpness, 1)
        self.assertEqual(sharp.path.sharpness, env.path.sharpness)
        np.testing.assert_array_equal(env.path.Gamma, sharp.path.Gamma)
        self.assertEqual(env.obs.b, 1.25)
        self.assertEqual(sharp.obs.b, 20)
        for values in log.values():
            self.assertTrue(np.isfinite(values).all())

    def test_measured_speed_and_final_frame(self):
        # Irregular times also expose accidental use of render stride or speed bounds.
        log = dict(t=np.array([0., 0.5, 2.]), x=np.array([[0., 0.], [3., 4.], [3., 7.]]))
        speed = measured_speed(log)
        self.assertTrue(np.isnan(speed[0]))
        np.testing.assert_array_equal(speed[1:], [10., 2.])
        np.testing.assert_array_equal(frame_indices(log, 10), [0, 2])
        np.testing.assert_array_equal(frame_indices(log, 1), [0, 1, 2])

    def test_destination_geometry_and_synchronized_nominal_arrival(self):
        env, prm, _, _ = self.runs["case5"]
        goal = env.path.w[-1]
        bot_distance = np.linalg.norm(goal - env.path.w[0])
        obstacle_distance = np.linalg.norm(goal - env.obs.c0[0])
        self.assertEqual(bot_distance, 120)
        self.assertEqual(bot_distance, obstacle_distance)
        self.assertEqual(prm.v_m, np.linalg.norm(env.obs.vel[0]))
        self.assertEqual(np.dot(goal - env.path.w[0], env.obs.vel[0]), 0)
        arrival = bot_distance / prm.v_m
        self.assertEqual(arrival, 24)
        np.testing.assert_array_equal(env.obs.centers(arrival)[0], goal)
        stationary, _, _, _ = self.runs["case4"]
        np.testing.assert_array_equal(stationary.obs.centers(0)[0], stationary.path.w[-1])
        np.testing.assert_array_equal(stationary.obs.vel[0], [0, 0])

    def test_position_goal_capture_rejects_projection_and_contact(self):
        env, _, _, _ = self.runs["case4"]
        self.assertTrue(goal_reached(env, [120, 10], 0, 2, "progress"))
        self.assertFalse(goal_reached(env, [120, 10], 0, 2, "position"))
        self.assertFalse(goal_reached(env, [120, 0], 0, 2, "position"))
        moving, _, _, _ = self.runs["case5"]
        self.assertFalse(goal_reached(moving, [120, 0], 24, 2, "position"))
        # The bot can be outside O1 while its goal region is still occupied.
        self.assertFalse(goal_reached(moving, [120, 1.9], 24.4, 2, "position"))
        self.assertTrue(goal_reached(moving, [120, 0], 26, 2, "position"))

    def test_destination_runs_stay_safe_and_report_actual_goal_outcomes(self):
        env, _, _, log = self.runs["case4"]
        self.assertGreater(log["t"][-1], 35)
        self.assertFalse(goal_reached(env, log["x"][-1], log["t"][-1], 2, "position"))
        self.assertGreater(min_clearance(log["x"], log["t"], env.obs).min(), 0)
        env, _, _, log = self.runs["case5"]
        self.assertGreater(log["t"][-1], 24)
        self.assertTrue(goal_reached(env, log["x"][-1], log["t"][-1], 2, "position"))
        self.assertGreater(min_clearance(log["x"], log["t"], env.obs).min(), 0)
        encounter = int(np.argmin(abs(log["t"] - 24)))
        self.assertGreater(np.linalg.norm(log["x"][encounter] - env.path.w[-1]), 3)

    def test_default_termination_is_progress_based(self):
        env, prm, _, log = self.runs["case2"]
        explicit = run_fppa(env, env.path.w[0], prm, t_final=200, goal_mode="progress")
        np.testing.assert_array_equal(log["x"], explicit["x"])
        np.testing.assert_array_equal(log["t"], explicit["t"])

    def test_renderer_matches_environment_and_tracks_physical_objects(self):
        for name in self.runs:
            env, _, _, log = self.runs[name]
            options = dict(goal_mode="position", goal_tol=2) if name in ("case4", "case5") else {}
            if name == "case5":
                options["nominal_speed"] = 5
            scene = AnimationScene(env, log, resolution=50, **options)
            try:
                k = min(700, len(log["t"]) - 1)
                scene.draw(k)
                t = log["t"][k]
                np.testing.assert_allclose(scene.image.get_array(), env.evaluate(scene.points, t)["mu_E"], atol=2e-15)
                for i, (ring, label) in enumerate(zip(scene.rings, scene.labels)):
                    np.testing.assert_array_equal(ring.center, env.obs.centers(t)[i])
                    self.assertEqual(ring.radius, env.obs.lam[i])
                    self.assertEqual(label.get_text(), f"O{i + 1}")
                    self.assertEqual(label.get_color(), RED)
                    self.assertIsNone(label.get_bbox_patch())
                self.assertIn(f"{measured_speed(log)[k]:.2f} m/s", scene.speed_tag.get_text())
                np.testing.assert_array_equal(scene.speed_tag.xy, log["x"][k])
                # The banner survives leaving an obstacle and repeated/out-of-order
                # draw calls cannot inflate the counter or carry it across loops.
                scene.draw(len(log["t"]) - 1)
                count = int(crash_counts(log, env.obs)[-1])
                expected = f"CRASHED · Crashes: {count}" if count else ""
                self.assertEqual(scene.crash_banner.get_text(), expected)
                self.assertEqual(scene.crash_banner.get_color(), RED)
                scene.draw(0)
                self.assertEqual(scene.crash_banner.get_text(), "")
                scene.draw(len(log["t"]) - 1)
                self.assertEqual(scene.crash_banner.get_text(), expected)
                if name == "case4":
                    self.assertIn("DESTINATION OCCUPIED", scene.status.get_text())
                if name == "case5":
                    self.assertIn("GOAL REACHED", scene.status.get_text())
                    np.testing.assert_array_equal(np.array(scene.nominal_bot.get_data()).ravel(), env.path.w[-1])
            finally:
                plt.close(scene.fig)


class CrashCounterTests(unittest.TestCase):
    def test_contact_episodes_reentry_and_multiple_objects(self):
        obs = FuzzyObstacles([[0, 0], [4, 0]], lam=1)
        log = dict(t=np.arange(8.), x=np.c_[[-3, 0, 0, -3, 0, 4, 4, 0], np.zeros(8)])
        np.testing.assert_array_equal(crash_counts(log, obs), [0, 1, 1, 1, 2, 3, 3, 4])
        overlap = FuzzyObstacles([[0, 0], [0.5, 0]], lam=1)
        log = dict(t=np.arange(3.), x=np.array([[0., 0.], [0., 0.], [3., 0.]]))
        np.testing.assert_array_equal(crash_counts(log, overlap), [2, 2, 2])
        np.testing.assert_array_equal(crash_counts(log, overlap, [np.array([0, 1])]), [1, 1, 1])

    def test_moving_object_contact_between_rendered_frames(self):
        obs = FuzzyObstacles([[-3, 0]], lam=1, vel=[2, 0])
        log = dict(t=np.arange(5.), x=np.zeros((5, 2)))
        np.testing.assert_array_equal(crash_counts(log, obs), [0, 1, 1, 1, 1])
        # A render stride of 10 skips both contact samples, but final count is 1.
        np.testing.assert_array_equal(crash_counts(log, obs)[frame_indices(log, 10)], [0, 1])


if __name__ == "__main__":
    unittest.main()
