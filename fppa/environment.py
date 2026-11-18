"""Fuzzy obstacles (Sec. 2.1), fuzzy path (Sec. 2.2) and fuzzy environment (Sec. 2.3)."""
import numpy as np

from .fuzzy import R90, SATURATIONS, bell_bmf, gauss_bmf, yager_with_grad


class FuzzyObstacles:
    """Atomic fuzzy obstacles, eqs. (2)-(3).

    centers: (N, 2) atom centres c_i at t = 0      lam: (N,) scales lambda_i
    r:       (N,)   safe distances r_i             vel: (N, 2) velocities v_i
    delta:   velocity-inflation gain               b:   BMF sharpness (eq. 6)
    Atoms move with constant velocity, c_i(t) = c_i + v_i t.
    """

    def __init__(self, centers, lam, r=0.0, vel=None, delta=0.0, b=4.0):
        self.c0 = np.atleast_2d(np.asarray(centers, float))
        n = len(self.c0)
        self.lam = np.broadcast_to(np.asarray(lam, float), (n,)).copy()
        self.r = np.broadcast_to(np.asarray(r, float), (n,)).copy()
        self.vel = np.zeros((n, 2)) if vel is None else np.broadcast_to(np.asarray(vel, float), (n, 2)).copy()
        self.delta, self.b = delta, b
        self.Linv = self._inverse_scale_matrices()

    def __len__(self):
        return len(self.c0)

    def scale_matrices(self):
        """Lambda_i = (lam_i + r_i) I + [q, q_perp] diag(delta |v_i|, 0) [q, q_perp]^T   (eq. 3)
                    = (lam_i + r_i) I + delta |v_i| q q^T."""
        speed = np.linalg.norm(self.vel, axis=1)
        q = np.where(speed[:, None] > 0, self.vel / np.maximum(speed, 1e-12)[:, None], 0.0)
        qqT = q[:, :, None] * q[:, None, :]
        return (self.lam + self.r)[:, None, None] * np.eye(2) + (self.delta * speed)[:, None, None] * qqT

    def _inverse_scale_matrices(self):
        # Lambda is symmetric with eigenpairs (a + d|v|, q), (a, q_perp), so
        # Lambda^-1 = (I - q q^T)/a + q q^T/(a + d|v|).  Plain inversion is fine for 2x2.
        return np.linalg.inv(self.scale_matrices())

    def centers(self, t=0.0):
        return self.c0 + self.vel * t

    def atoms(self, X, t=0.0):
        """Memberships of every atom: mu_i(x) = mu(Lambda_i^-1 (x - c_i))  (eq. 2).
        X: (..., 2) -> mu (..., N), grad (..., N, 2).  Lambda symmetric => grad_x = Lambda^-1 grad_chi."""
        d = X[..., None, :] - self.centers(t)                  # (..., N, 2)
        chi = np.einsum("nij,...nj->...ni", self.Linv, d)
        mu, g_chi = bell_bmf(chi, self.b)
        return mu, np.einsum("nij,...nj->...ni", self.Linv, g_chi)


class FuzzyPath:
    """Chain of Gaussian fuzzy waypoints, eqs. (8)-(10).

    For each segment w_j -> w_{j+1}:  p_j = (w_j + w_{j+1})/2,
        Gamma_j = H_j diag(gamma |w_{j+1} - w_j|, sigma_j) H_j^T,  H_j = [h_j, h_j_perp].
    (The paper prints [h, h_perp]^T D [h, h_perp]; with h as *columns* the matrix with
    eigenvector h is H D H^T, consistent with eq. (3).)
    """

    def __init__(self, waypoints, sigma=1.0, gamma=0.75):
        self.w = np.asarray(waypoints, float)
        seg = np.diff(self.w, axis=0)
        L = np.linalg.norm(seg, axis=1)
        h = seg / L[:, None]
        H = np.stack([h, h @ R90.T], axis=-1)                    # columns h, h_perp
        m = len(seg)
        sig = np.broadcast_to(np.asarray(sigma, float), (m,))
        D = np.zeros((m, 2, 2))
        D[:, 0, 0], D[:, 1, 1] = gamma * L, sig
        self.p = 0.5 * (self.w[:-1] + self.w[1:])
        self.Gamma = H @ D @ np.transpose(H, (0, 2, 1))
        self.Ginv = np.linalg.inv(self.Gamma)
        self.cum = np.concatenate([[0.0], np.cumsum(L)])         # arc length at waypoints

    @property
    def length(self):
        return self.cum[-1]

    def atoms(self, X):
        """mu_Wj(x) = xi(Gamma_j^-1 (x - p_j))  (eq. 9).  grad_x = Gamma^-1 grad_chi (Gamma symmetric)."""
        chi = np.einsum("nij,...nj->...ni", self.Ginv, X[..., None, :] - self.p)
        mu, g_chi = gauss_bmf(chi)
        return mu, np.einsum("nij,...nj->...ni", self.Ginv, g_chi)

    def progress(self, x):
        """Arc-length coordinate of the projection of x onto the crisp polyline."""
        a, b = self.w[:-1], self.w[1:]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", x - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
        proj = a + t[:, None] * ab
        i = np.argmin(np.linalg.norm(proj - x, axis=1))
        return self.cum[i] + t[i] * np.linalg.norm(ab[i]), np.linalg.norm(proj[i] - x)


class FuzzyEnvironment:
    """Environment episode E^k = O^k  U  P with Yager unions (eqs. 4-5, 11-16).

    sat:  'tanh' (paper), 'exact' (min(1, .)), or 'softmin'.
    flat: the smoothed operator is *not* associative, so we must choose how to
          nest it.  flat=True evaluates E as one m-ary union over all atoms (what
          Remark 1 suggests); flat=False evaluates sat(mu_O, mu_P) literally.
    """

    def __init__(self, obstacles, path, p=1.0, sat="tanh", flat=True):
        self.obs, self.path, self.p, self.flat = obstacles, path, p, flat
        self.sat = SATURATIONS[sat] if isinstance(sat, str) else sat

    def evaluate(self, X, t=0.0):
        X = np.asarray(X, float)
        mP_i, gP_i = self.path.atoms(X)
        mu_P, g_P = yager_with_grad(mP_i, gP_i, self.p, self.sat)
        if self.obs is None or len(self.obs) == 0:
            zeros = np.zeros(X.shape[:-1])
            return dict(mu_E=mu_P, g_E=g_P, mu_O=zeros, g_O=np.zeros_like(X), mu_P=mu_P)
        mO_i, gO_i = self.obs.atoms(X, t)
        mu_O, g_O = yager_with_grad(mO_i, gO_i, self.p, self.sat)
        if self.flat:
            mu_E, g_E = yager_with_grad(np.concatenate([mO_i, mP_i], -1),
                                        np.concatenate([gO_i, gP_i], -2), self.p, self.sat)
        else:
            mu_E, g_E = yager_with_grad(np.stack([mu_O, mu_P], -1), np.stack([g_O, g_P], -2),
                                        self.p, self.sat)
        return dict(mu_E=mu_E, g_E=g_E, mu_O=mu_O, g_O=g_O, mu_P=mu_P)
