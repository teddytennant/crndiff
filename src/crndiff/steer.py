"""Marginal tilt and Feynman-Kac particle correction from arXiv:2609.31149.

Tilted posterior (paper):

    Q^H(x_0 | x_t, t) ∝ H(x_0) Q(x_0 | x_t, t)

with coordinate-wise tilt H(n) = prod_i u_i(n_i) and

    u_i(n_i) = (D_{R,i}(n_i) / D_{theta,i}(n_i))^tau.

FK increment (paper):

    G_j = psi_{j-1}(x_{t_{j-1}}) / psi_j(x_{t_j}),
    psi_K ≡ 1,  psi_0 = rho,  psi_j(x_{t_j}) = E[rho(x_0) | x_{t_j}].

Tilting is applied to Q before the FK weight correction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crndiff.ffbs import posterior_logits, sample_bridge
from crndiff.kernel import K_MINUS, MAX_COUNT


def coordinate_log_marginals(log_pmf, grid, n_max: int):
    """log D_i(k) for k = 0..n_max from a joint log-pmf on a count grid."""
    k = jnp.arange(n_max + 1)
    eq = grid[:, :, None] == k[None, None, :]
    masked = jnp.where(eq, log_pmf[:, None, None], -jnp.inf)
    return jax.nn.logsumexp(masked, axis=0)


def log_marginal_tilt(log_D, log_D_ref, grid, n_max: int, tau: float = 1.0, floor: float = 1e-12):
    """log H(n) = tau * sum_i (log D_{R,i} - log D_{theta,i}) on the grid."""
    log_m_gen = coordinate_log_marginals(log_D, grid, n_max)
    log_m_ref = coordinate_log_marginals(log_D_ref, grid, n_max)
    log_u = tau * (log_m_ref - jnp.maximum(log_m_gen, jnp.log(floor)))
    idx = grid.astype(jnp.int32)
    log_h = jnp.zeros(grid.shape[0])
    for i in range(grid.shape[1]):
        log_h = log_h + log_u[i, idx[:, i]]
    return log_h, log_u


def tilt_posterior_logits(log_q, log_h):
    """Q^H ∝ H Q, as written."""
    logits = log_q + log_h
    return logits - jax.nn.logsumexp(logits)


def continuation_potential(log_q, rho):
    """psi(x_t) = E[rho(x_0) | x_t] on the finite grid."""
    q = jnp.exp(log_q)
    return jnp.sum(q * rho)


def ess(weights):
    w = jnp.asarray(weights)
    return (jnp.sum(w) ** 2) / jnp.sum(w ** 2)


def importance_ess_frac(log_q, log_target):
    """Large-M ESS/M for importance weights target/q, both log-normalized on a grid."""
    # (E_q w)^2 / E_q w^2 with w = target/q. If both pmfs, E_q w = 1.
    ew2 = jnp.sum(jnp.exp(2.0 * log_target - log_q))
    return 1.0 / ew2


def systematic_resample(key, weights):
    w = jnp.asarray(weights)
    w = w / jnp.sum(w)
    n = w.shape[0]
    u0 = jax.random.uniform(key, minval=0.0, maxval=1.0 / n)
    positions = u0 + jnp.arange(n) / n
    indices = jnp.searchsorted(jnp.cumsum(w), positions, side="left")
    return jnp.clip(indices, 0, n - 1)


def _init_terminal(key, n_particles, mu, n_max):
    x = jax.random.poisson(key, mu, shape=(n_particles, mu.shape[0]))
    return jnp.clip(x, 0, n_max).astype(jnp.int32)


def tilted_fk_sample(
    key,
    n_particles: int,
    times,
    mu,
    grid,
    log_D,
    log_D_ref,
    n_max: int,
    tau: float = 1.0,
    resample_ess_frac: float = 0.5,
    k_minus: float = K_MINUS,
    max_m: int = MAX_COUNT,
):
    """Tilt posterior marginals, then FK-correct with rho = D_ref / D_tilt.

    Returns (x0, weights, ess_history).
    """
    times = jnp.asarray(times)
    mu = jnp.asarray(mu)
    n_steps = int(times.shape[0]) - 1
    log_h, _ = log_marginal_tilt(log_D, log_D_ref, grid, n_max, tau=tau)
    log_D_tilt = log_D + log_h
    log_D_tilt = log_D_tilt - jax.nn.logsumexp(log_D_tilt)
    log_rho = log_D_ref - log_D_tilt
    rho = jnp.exp(log_rho)

    key, k_init = jax.random.split(key)
    x = _init_terminal(k_init, n_particles, mu, n_max)
    w = jnp.ones((n_particles,))
    x0_out = x
    ess_hist = []

    for j in range(n_steps, 0, -1):
        t_t = times[j]
        t_s = times[j - 1]
        key, k_step = jax.random.split(key)
        keys = jax.random.split(k_step, n_particles)

        def logits_at(xt):
            return posterior_logits(
                xt,
                t_t,
                mu,
                grid,
                log_D,
                log_H=log_h,
                k_minus=k_minus,
                max_m=max_m,
            )

        def psi_at(xt):
            return continuation_potential(logits_at(xt), rho)

        psi_t = jax.vmap(psi_at)(x)
        psi_t = jnp.maximum(psi_t, 1e-12)

        def one(k, xt):
            logits = logits_at(xt)
            k1, k2 = jax.random.split(k)
            i0 = jax.random.categorical(k1, logits)
            x0 = grid[i0].astype(jnp.int32)
            xs = sample_bridge(
                k2, x0, xt, t_s, t_t, mu, n_max, k_minus=k_minus, max_m=max_m
            )
            return xs, x0

        xs, x0 = jax.vmap(one)(keys, x)
        if j == 1:
            # psi_0 = rho(x_0)
            idx = _grid_index(x0, n_max)
            psi_s = rho[idx]
        else:
            psi_s = jax.vmap(psi_at)(xs)
        psi_s = jnp.maximum(psi_s, 1e-12)
        g = psi_s / psi_t
        w = w * g
        x = xs
        x0_out = x0
        ess_hist.append(ess(w))
        if ess(w) < resample_ess_frac * n_particles:
            key, k_rs = jax.random.split(key)
            idx = systematic_resample(k_rs, w)
            x = x[idx]
            x0_out = x0_out[idx]
            w = jnp.ones((n_particles,))

    return x0_out.astype(jnp.int32), w, jnp.stack(ess_hist)


def _grid_index(n, n_max: int):
    """Row-major index of a count vector in {0,...,n_max}^S."""
    n = n.astype(jnp.int32)
    s = n.shape[-1]
    idx = jnp.zeros(n.shape[:-1], dtype=jnp.int32)
    stride = 1
    for i in range(s - 1, -1, -1):
        idx = idx + n[..., i] * stride
        stride = stride * (n_max + 1)
    return idx


def naive_importance_weights(log_D, log_D_ref):
    """Endpoint importance rho = D_ref / D on the generator (untilted)."""
    return jnp.exp(log_D_ref - log_D)
