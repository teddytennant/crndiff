"""Data-driven terminal noising time T_O from arXiv:2609.31149.

Displayed criterion:

    sigma_noise = (1 + sqrt(S / N))^2 - 1     (Marchenko-Pastur bulk edge)

    T_O = (1/2) log(sigma_1 / sigma_noise)    if sigma_1 > sigma_noise,
          0                                   otherwise.

sigma_1 is the largest generalized eigenvalue of (Psi_0, mud_0), where
Psi_{0,ij} = Cov(n_i, n_j) - delta_{ij} E[n_i] and mud_0 = diag(E[n]).
Moments are under the empirical measure D (divide by N).
"""

from __future__ import annotations

import jax.numpy as jnp


def excess_covariance(counts):
    """Psi_0 and mu from an (N, S) count array under the empirical measure."""
    counts = jnp.asarray(counts)
    n_obs = counts.shape[0]
    mu = jnp.mean(counts, axis=0)
    centered = counts - mu
    cov = (centered.T @ centered) / n_obs
    psi = cov - jnp.diag(mu)
    return psi, mu


def mp_edge(n_obs: int, n_species: int):
    """sigma_noise = (1 + sqrt(S/N))^2 - 1."""
    ratio = jnp.sqrt(jnp.asarray(n_species) / jnp.asarray(n_obs))
    return (1.0 + ratio) ** 2 - 1.0


def largest_generalized_eigenvalue(psi, mu):
    """Largest generalized eigenvalue of (Psi, diag(mu)): Psi v = sigma diag(mu) v."""
    psi = jnp.asarray(psi)
    mu = jnp.asarray(mu)
    mu_safe = jnp.maximum(mu, 1e-12)
    inv_sqrt = 1.0 / jnp.sqrt(mu_safe)
    # C = mud^{-1/2} Psi mud^{-1/2}
    c = inv_sqrt[:, None] * psi * inv_sqrt[None, :]
    c = 0.5 * (c + c.T)
    evals = jnp.linalg.eigvalsh(c)
    return evals[-1]


def terminal_noising_time(counts):
    """T_O from the displayed formula. Deterministic given counts."""
    counts = jnp.asarray(counts)
    n_obs, n_species = counts.shape
    psi, mu = excess_covariance(counts)
    sigma_1 = largest_generalized_eigenvalue(psi, mu)
    sigma_noise = mp_edge(n_obs, n_species)
    log_arg = sigma_1 / sigma_noise
    t_o = 0.5 * jnp.log(log_arg)
    t_o = jnp.where(sigma_1 > sigma_noise, t_o, jnp.zeros_like(t_o))
    t_o = jnp.maximum(t_o, 0.0)
    return t_o, sigma_1, sigma_noise
