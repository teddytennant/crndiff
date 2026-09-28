"""Independent birth-death CRN forward kernel from arXiv:2609.31149.

Displayed kernel (independent birth-death instantiation):

    q(n_t | n_0) = prod_i [Bin(n_{0,i}, beta_{t,i}) * Pois(mu_i (1 - beta_{t,i}))](n_{t,i})

    = prod_i sum_{m=0}^{n_{t,i}} Bin(n_{0,i}, beta_{t,i})(m)
                                 * Pois(mu_i (1 - beta_{t,i}))(n_{t,i} - m)

with beta_{t,i} = exp(-k_i^- t) and mu_i = k_i^+ / k_i^-.
The paper sets k_i^- = 1 and k_i^+ = mu_i = E_D[n_i], which preserves means.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlog1py, xlogy

# Paper: k_i^- = 1 for all coordinates.
K_MINUS = 1.0
# Convolution is a finite sum over survivors m = 0..min(n_0, n_t); this caps n_0.
MAX_COUNT = 64


def beta_t(t, k_minus: float = K_MINUS):
    """Survival probability beta_{t,i} = exp(-k_i^- t)."""
    return jnp.exp(-k_minus * jnp.asarray(t))


def birth_death_rates(mu, k_minus: float = K_MINUS):
    """Paper choice: k^- = 1, k^+ = mu, so the Poisson equilibrium has mean mu."""
    mu = jnp.asarray(mu)
    k_minus_v = jnp.full_like(mu, k_minus, dtype=mu.dtype)
    k_plus = mu * k_minus_v
    return k_plus, k_minus_v


def log_binom_pmf(k, n, p):
    """log Bin(n, p)(k), broadcastable. Paper uses Bin(n_{0,i}, beta_{t,i})."""
    k = jnp.asarray(k)
    n = jnp.asarray(n)
    p = jnp.asarray(p)
    k_f = k.astype(p.dtype)
    n_f = n.astype(p.dtype)
    k_safe = jnp.clip(k_f, 0.0, jnp.maximum(n_f, 0.0))
    logc = gammaln(n_f + 1.0) - gammaln(k_safe + 1.0) - gammaln(n_f - k_safe + 1.0)
    logp = logc + xlogy(k_safe, p) + xlog1py(n_f - k_safe, -p)
    valid = (k_f >= 0.0) & (k_f <= n_f)
    return jnp.where(valid, logp, -jnp.inf)


def log_poisson_pmf(k, lam):
    """log Pois(lam)(k), broadcastable."""
    k = jnp.asarray(k)
    lam = jnp.asarray(lam)
    k_f = k.astype(lam.dtype)
    k_safe = jnp.clip(k_f, 0.0, jnp.inf)
    logp = -lam + xlogy(k_safe, lam) - gammaln(k_safe + 1.0)
    return jnp.where(k_f >= 0.0, logp, -jnp.inf)


def log_binpois_conv(n_t, n_0, beta, lam, max_m: int = MAX_COUNT):
    """log of the displayed convolution [Bin(n_0, beta) * Pois(lam)](n_t)."""
    n_t = jnp.asarray(n_t)
    n_0 = jnp.asarray(n_0)
    beta = jnp.asarray(beta)
    lam = jnp.asarray(lam)
    shape = jnp.broadcast_shapes(n_t.shape, n_0.shape, beta.shape, lam.shape)
    n_t = jnp.broadcast_to(n_t, shape)
    n_0 = jnp.broadcast_to(n_0, shape)
    beta = jnp.broadcast_to(beta, shape)
    lam = jnp.broadcast_to(lam, shape)
    m = jnp.arange(max_m + 1)
    m_b = m.reshape((max_m + 1,) + (1,) * n_t.ndim).astype(n_t.dtype)
    log_terms = log_binom_pmf(m_b, n_0, beta) + log_poisson_pmf(n_t - m_b, lam)
    return jax.nn.logsumexp(log_terms, axis=0)


def log_forward_kernel(n_t, n_0, t, mu, k_minus: float = K_MINUS, max_m: int = MAX_COUNT):
    """log q(n_t | n_0) as a product of independent birth-death convolutions.

    n_t, n_0: (..., S) nonnegative integers.
    t: scalar or broadcastable to n_t.shape[:-1].
    mu: (S,) equilibrium means.
    """
    n_t = jnp.asarray(n_t)
    n_0 = jnp.asarray(n_0)
    mu = jnp.asarray(mu)
    t = jnp.asarray(t)
    beta = beta_t(t, k_minus)
    if beta.ndim == n_t.ndim - 1:
        beta = jnp.expand_dims(beta, axis=-1)
    lam = mu * (1.0 - beta)
    log_coord = log_binpois_conv(n_t, n_0, beta, lam, max_m=max_m)
    return jnp.sum(log_coord, axis=-1)


def forward_kernel_pmf_1d(
    n_0, t, mu, n_max: int, k_minus: float = K_MINUS, max_m: int = MAX_COUNT
):
    """p(n_t = 0, ..., n_max | n_0) for one coordinate."""
    n_t = jnp.arange(n_max + 1)
    n_0 = jnp.asarray(n_0)
    beta = beta_t(t, k_minus)
    lam = jnp.asarray(mu) * (1.0 - beta)
    return jnp.exp(log_binpois_conv(n_t, n_0, beta, lam, max_m=max_m))


def sample_forward(key, n_0, t, mu, k_minus: float = K_MINUS):
    """Sample n_t | n_0 from the closed-form birth-death kernel.

    Survivors ~ Bin(n_0, beta_t), births ~ Pois(mu (1 - beta_t)).
    """
    n_0 = jnp.asarray(n_0)
    mu = jnp.asarray(mu)
    t = jnp.asarray(t)
    beta = beta_t(t, k_minus)
    if beta.ndim == n_0.ndim - 1:
        beta = jnp.expand_dims(beta, axis=-1)
    lam = mu * (1.0 - beta)
    k1, k2 = jax.random.split(key)
    n_trials = n_0.astype(jnp.int32)
    survivors = jax.random.binomial(k1, n_trials, beta.astype(n_0.dtype))
    births = jax.random.poisson(k2, lam)
    return (survivors + births).astype(jnp.int32)
