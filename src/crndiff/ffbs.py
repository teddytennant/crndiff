"""Forward-filtering backward-sampling using the closed-form birth-death kernel.

On 0 = t_0 < ... < t_K = T:

    x_K ~ pi
    for j = K, ..., 1:
        x_0 ~ Q(x_0 | x_{t_j}, t_j)
        x_{t_{j-1}} ~ B(x_{t_{j-1}} | x_0, x_{t_j})

with bridge

    B(x_s | x_0, x_t) = q_{t-s}(x_t | x_s) q_s(x_s | x_0) / q_t(x_t | x_0).

Q is the exact finite-grid posterior of a known count-space D (no neural net).
The birth-death kernel factorizes, so the bridge is sampled coordinate-wise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crndiff.kernel import (
    K_MINUS,
    MAX_COUNT,
    beta_t,
    log_binpois_conv,
    log_forward_kernel,
)


def log_bridge(n_s, n_0, n_t, t_s, t_t, mu, k_minus: float = K_MINUS, max_m: int = MAX_COUNT):
    """log B(n_s | n_0, n_t) from the displayed bridge, product over coordinates."""
    log_num = log_forward_kernel(
        n_t, n_s, t_t - t_s, mu, k_minus=k_minus, max_m=max_m
    ) + log_forward_kernel(n_s, n_0, t_s, mu, k_minus=k_minus, max_m=max_m)
    log_den = log_forward_kernel(n_t, n_0, t_t, mu, k_minus=k_minus, max_m=max_m)
    return log_num - log_den


def _sample_bridge_1d(key, n_0, n_t, t_s, t_t, mu_i, n_max, k_minus, max_m):
    n_s = jnp.arange(n_max + 1)
    beta_s = beta_t(t_s, k_minus)
    beta_dt = beta_t(t_t - t_s, k_minus)
    beta_t_tot = beta_t(t_t, k_minus)
    log_q_t_from_s = log_binpois_conv(
        n_t, n_s, beta_dt, mu_i * (1.0 - beta_dt), max_m=max_m
    )
    log_q_s_from_0 = log_binpois_conv(
        n_s, n_0, beta_s, mu_i * (1.0 - beta_s), max_m=max_m
    )
    log_q_t_from_0 = log_binpois_conv(
        n_t, n_0, beta_t_tot, mu_i * (1.0 - beta_t_tot), max_m=max_m
    )
    logits = log_q_t_from_s + log_q_s_from_0 - log_q_t_from_0
    drawn = jax.random.categorical(key, logits)
    drawn = jnp.where(t_s <= 0, jnp.asarray(n_0, dtype=drawn.dtype), drawn)
    drawn = jnp.where(t_s >= t_t, jnp.asarray(n_t, dtype=drawn.dtype), drawn)
    return drawn.astype(jnp.int32)


def sample_bridge(
    key,
    n_0,
    n_t,
    t_s,
    t_t,
    mu,
    n_max: int,
    k_minus: float = K_MINUS,
    max_m: int = MAX_COUNT,
):
    """Sample n_s | n_0, n_t independently per coordinate."""
    n_0 = jnp.asarray(n_0).astype(jnp.int32)
    n_t = jnp.asarray(n_t).astype(jnp.int32)
    mu = jnp.asarray(mu)
    n_species = n_0.shape[-1]
    keys = jax.random.split(key, n_species)

    def one(k, n0i, nti, mui):
        return _sample_bridge_1d(k, n0i, nti, t_s, t_t, mui, n_max, k_minus, max_m)

    return jax.vmap(one)(keys, n_0, n_t, mu)


def posterior_logits(
    n_t,
    t,
    mu,
    grid,
    log_D,
    log_H=None,
    k_minus: float = K_MINUS,
    max_m: int = MAX_COUNT,
):
    """log Q(n_0 | n_t, t) on a finite count grid, optionally tilted by H(n_0).

    Q(n_0 | n_t) ∝ q(n_t | n_0) D(n_0), and the tilted posterior is
    Q^H(n_0 | n_t) ∝ H(n_0) Q(n_0 | n_t)  (paper, tilt_posterior).
    """
    n_t = jnp.asarray(n_t)
    log_lik = jax.vmap(
        lambda n0: log_forward_kernel(n_t, n0, t, mu, k_minus=k_minus, max_m=max_m)
    )(grid)
    logits = log_lik + log_D
    if log_H is not None:
        logits = logits + log_H
    return logits - jax.nn.logsumexp(logits)


def _init_terminal(key, n_particles, mu, n_max):
    x = jax.random.poisson(key, mu, shape=(n_particles, mu.shape[0]))
    return jnp.clip(x, 0, n_max).astype(jnp.int32)


def ffbs_sample(
    key,
    n_particles: int,
    times,
    mu,
    grid,
    log_D,
    n_max: int,
    log_H=None,
    k_minus: float = K_MINUS,
    max_m: int = MAX_COUNT,
):
    """Unconditional (or marginally tilted) FFBS reverse sample of x_0.

    times[0] = 0, times[-1] = T. Initialization is the product-Poisson
    equilibrium pi as in the paper.
    """
    times = jnp.asarray(times)
    mu = jnp.asarray(mu)
    n_steps = int(times.shape[0]) - 1
    key, k_init = jax.random.split(key)
    x = _init_terminal(k_init, n_particles, mu, n_max)
    x0_out = x

    def body(j, carry):
        key, x, x0_out = carry
        t_t = times[j]
        t_s = times[j - 1]
        key, k_step = jax.random.split(key)
        keys = jax.random.split(k_step, n_particles)

        def one(k, xt):
            logits = posterior_logits(
                xt, t_t, mu, grid, log_D, log_H=log_H, k_minus=k_minus, max_m=max_m
            )
            k1, k2 = jax.random.split(k)
            i0 = jax.random.categorical(k1, logits)
            x0 = grid[i0].astype(jnp.int32)
            xs = sample_bridge(
                k2, x0, xt, t_s, t_t, mu, n_max, k_minus=k_minus, max_m=max_m
            )
            return xs, x0

        xs, x0 = jax.vmap(one)(keys, x)
        return key, xs, x0

    # Python loop over a tiny reverse schedule so tests stay readable.
    for j in range(n_steps, 0, -1):
        key, x, x0_out = body(j, (key, x, x0_out))
    return x0_out.astype(jnp.int32)
