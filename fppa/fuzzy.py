"""Fuzzy-set primitives used by the FPPA (Rastkhiz, Schwartz, Lambadaris, FSS 2025).

Every function returns the membership value *and* its gradient, because the
planner (eq. 22) is a gradient method and we want exact, not finite-difference,
derivatives.

Shapes: points are arrays of shape (..., 2); gradients have shape (..., 2).
"""
import numpy as np

R90 = np.array([[0.0, -1.0], [1.0, 0.0]])  # R(pi/2), rotation by +90 degrees

_S_MIN = 1e-9  # guards chi^T chi -> 0 in the bell BMF


# ---------------------------------------------------------------------------
# Base membership functions (BMFs)
# ---------------------------------------------------------------------------
def bell_bmf(chi, b):
    """Obstacle BMF, eq. (6):  mu(chi) = tanh((chi^T chi)^(-b)).

    Returns (mu, grad_chi).  With s = chi^T chi and u = s^(-b):
        d mu / d chi = sech^2(u) * (-b) * s^(-b-1) * 2 chi
    """
    s = np.maximum(np.sum(chi * chi, axis=-1), _S_MIN)
    # tanh is already exactly 1 at u=20 in float64. Cap there before
    # exponentiating so steep memberships cannot produce 0 * inf gradients.
    u = np.exp(np.minimum(-b * np.log(s), np.log(20.0)))
    mu = np.tanh(u)
    sech2 = 1.0 - mu * mu  # exactly 0 once tanh saturates, avoids 0*inf
    coef = -2.0 * b * sech2 * u / s
    return mu, coef[..., None] * chi


def gauss_bmf(chi, sharpness=1.0):
    """Path BMF exp(-s**k / 2), s=chi^T chi; k=1 is eq. (8).

    k>1 gives a smooth near-crisp ellipse. Evaluate powers and derivatives in
    log space; outside the representable tail both value and gradient are zero.
    """
    if not np.isfinite(sharpness) or sharpness < 1:
        raise ValueError("path sharpness must be finite and >= 1")
    s = np.sum(chi * chi, axis=-1)
    if sharpness == 1:
        xi = np.exp(-0.5 * s)
        return xi, -xi[..., None] * chi
    log_s = np.log(np.maximum(s, np.finfo(float).tiny))
    log_u = sharpness * log_s
    active = log_u < np.log(1500.0)
    u = np.exp(np.minimum(log_u, np.log(1500.0)))
    xi = np.where(active, np.exp(-0.5 * u), 0.0)
    log_coef = np.log(sharpness) + (sharpness - 1) * log_s - 0.5 * u
    coef = np.exp(np.minimum(log_coef, 700.0))
    coef = np.where(active & (s > 0), coef, 0.0)
    return xi, -coef[..., None] * chi


# ---------------------------------------------------------------------------
# Saturations and Yager's class of t-conorms
# ---------------------------------------------------------------------------
def sat_exact(u):
    """sat(u) = min(1, u) and its (a.e.) derivative."""
    return np.minimum(1.0, u), (u < 1.0).astype(float)


def sat_tanh(u):
    """The paper's smooth surrogate of sat: tanh(u).  Note tanh(u) < u for u > 0,
    so the smoothed operator violates the t-conorm boundary law S(a, 0) = a."""
    t = np.tanh(u)
    return t, 1.0 - t * t


def make_sat_softmin(kappa=25.0):
    """Smooth min(1, u) via softplus:  u - log(1 + exp(kappa (u - 1))) / kappa,
    renormalised so that sat(0) = 0.  Converges uniformly to min(1, u) as
    kappa -> inf, and (unlike tanh) is ~identity on [0, 1 - O(1/kappa)]."""
    c0 = np.log1p(np.exp(-kappa)) / kappa

    def sat(u):
        z = kappa * (u - 1.0)
        val = u - np.logaddexp(0.0, z) / kappa + c0
        der = 1.0 - 1.0 / (1.0 + np.exp(-z))
        return val, der

    return sat


SATURATIONS = {"exact": sat_exact, "tanh": sat_tanh, "softmin": make_sat_softmin()}


def yager(a, p, sat=sat_exact, axis=-1):
    """Multi-operand Yager t-conorm, Remark 1 / eq. (7):
        S(a_1..a_m) = sat( (sum a_i^p)^(1/p) ),   p in [1, inf].
    p = np.inf gives the max t-conorm."""
    a = np.asarray(a, dtype=float)
    if np.isinf(p):
        return np.max(a, axis=axis)
    return sat((np.sum(a ** p, axis=axis)) ** (1.0 / p))[0]


def yager_with_grad(mus, grads, p, sat):
    """Yager union of m membership functions with chain-rule gradient.

    mus:   (..., m)      membership values of the operands at the points
    grads: (..., m, 2)   their spatial gradients
    Returns (mu, grad) of  sat(n),  n = ||a||_p.
        d n / d a_i = n^(1-p) a_i^(p-1)
    """
    mus = np.maximum(mus, 0.0)  # smooth sats can return -1e-17 near 0
    if p == 1:
        n = np.sum(mus, axis=-1)
        dn = np.sum(grads, axis=-2)
    else:
        # scale by the largest operand so that a^p cannot underflow (as in hypot)
        m = np.maximum(np.max(mus, axis=-1, keepdims=True), 1e-300)
        n = m[..., 0] * np.sum((mus / m) ** p, axis=-1) ** (1.0 / p)
        w = (mus / np.maximum(n, 1e-300)[..., None]) ** (p - 1.0)  # = n^(1-p) a^(p-1), in [0, 1]
        dn = np.sum(w[..., None] * grads, axis=-2)
    val, der = sat(n)
    return val, der[..., None] * dn
