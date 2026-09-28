"""Synthetic two-class count mixture (not cells)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crndiff.kernel import log_poisson_pmf


def two_class_params():
    """Majority class near (1, 7), rare target class near (7, 1)."""
    class_mu = jnp.array([[1.0, 7.0], [7.0, 1.0]])
    mix_weights = jnp.array([0.85, 0.15])
    return class_mu, mix_weights


def two_class_samples(key, n_obs: int, class_mu=None, mix_weights=None):
    if class_mu is None or mix_weights is None:
        class_mu, mix_weights = two_class_params()
    k1, k2 = jax.random.split(key)
    labels = jax.random.categorical(k1, jnp.log(mix_weights), shape=(n_obs,))
    mu = class_mu[labels]
    counts = jax.random.poisson(k2, mu)
    return counts.astype(jnp.int32), labels


def make_count_grid(n_max: int, n_species: int):
    axes = [jnp.arange(n_max + 1) for _ in range(n_species)]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([m.reshape(-1) for m in mesh], axis=-1).astype(jnp.int32)


def log_mixture_on_grid(grid, class_mu, mix_weights):
    """log D(n) for a product-Poisson mixture on a count grid."""
    log_comp = jnp.log(mix_weights)[:, None] + jnp.sum(
        log_poisson_pmf(grid[None, :, :], class_mu[:, None, :]), axis=-1
    )
    log_p = jax.nn.logsumexp(log_comp, axis=0)
    # Truncation onto the grid: renormalize.
    return log_p - jax.nn.logsumexp(log_p)


def class_log_probs(n, class_mu, mix_weights):
    """log p(n | class) and log p(class | n) for a product-Poisson mixture."""
    n = jnp.asarray(n)
    log_lik = jnp.sum(log_poisson_pmf(n[..., None, :], class_mu), axis=-1)
    log_joint = log_lik + jnp.log(mix_weights)
    log_marg = jax.nn.logsumexp(log_joint, axis=-1, keepdims=True)
    log_post = log_joint - log_marg
    return log_lik, log_post


def target_class_mass(n, class_mu=None, mix_weights=None):
    """Mean posterior mass on the rare (target) class."""
    if class_mu is None or mix_weights is None:
        class_mu, mix_weights = two_class_params()
    _, log_post = class_log_probs(n, class_mu, mix_weights)
    return jnp.mean(jnp.exp(log_post[..., 1]))
