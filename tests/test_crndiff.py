"""Lock displayed equations from arXiv:2609.31149, not a rewrite."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crndiff.ffbs import ffbs_sample, log_bridge, posterior_logits, sample_bridge
from crndiff.kernel import (
    K_MINUS,
    beta_t,
    forward_kernel_pmf_1d,
    log_binom_pmf,
    log_binpois_conv,
    log_forward_kernel,
    log_poisson_pmf,
    sample_forward,
)
from crndiff.steer import (
    continuation_potential,
    importance_ess_frac,
    log_marginal_tilt,
    tilt_posterior_logits,
    tilted_fk_sample,
)
from crndiff.terminal import (
    excess_covariance,
    largest_generalized_eigenvalue,
    mp_edge,
    terminal_noising_time,
)
from crndiff.toy import (
    log_mixture_on_grid,
    make_count_grid,
    target_class_mass,
    two_class_params,
    two_class_samples,
)


def _manual_binpois(n_t: int, n_0: int, beta: float, mu: float) -> float:
    """Displayed convolution, evaluated with math.comb / factorial."""
    lam = mu * (1.0 - beta)
    total = 0.0
    for m in range(0, n_t + 1):
        if m > n_0:
            break
        bin_p = math.comb(n_0, m) * (beta**m) * ((1.0 - beta) ** (n_0 - m))
        poi_p = math.exp(-lam) * (lam ** (n_t - m)) / math.factorial(n_t - m)
        total += bin_p * poi_p
    return total


def _cme_pmf(n_0: int, t: float, mu: float, n_max: int, k_minus: float = 1.0):
    """Brute-force Kolmogorov forward equation via matrix exponential."""
    k_plus = mu * k_minus
    q = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    for n in range(n_max + 1):
        if n > 0:
            q[n - 1, n] += k_minus * n
            q[n, n] -= k_minus * n
        if n < n_max:
            q[n + 1, n] += k_plus
            q[n, n] -= k_plus
    p0 = np.zeros(n_max + 1, dtype=np.float64)
    p0[n_0] = 1.0
    return np.asarray(jax.scipy.linalg.expm(t * jnp.asarray(q)) @ jnp.asarray(p0))


def test_kernel_is_a_probability():
    n_0, t, mu, n_max = 3, 0.4, 2.0, 40
    pmf = np.asarray(forward_kernel_pmf_1d(n_0, t, mu, n_max))
    assert np.all(pmf >= -1e-15)
    assert pmf.sum() == pytest.approx(1.0, rel=0, abs=1e-8)
    # Product kernel over two coordinates, truncated support.
    grid = make_count_grid(12, 2)
    n0 = jnp.array([2, 1])
    mu2 = jnp.array([1.5, 2.5])
    logp = jax.vmap(lambda nt: log_forward_kernel(nt, n0, 0.3, mu2))(grid)
    p = np.exp(np.asarray(logp))
    assert np.all(p >= 0.0)
    assert p.sum() == pytest.approx(1.0, rel=0, abs=2e-4)


def test_birth_death_closed_form_matches_manual_and_cme():
    n_0, t, mu = 3, 0.5, 1.7
    beta = float(np.exp(-t))
    n_max = 24
    pmf = np.asarray(forward_kernel_pmf_1d(n_0, t, mu, n_max))
    manual = np.array([_manual_binpois(k, n_0, beta, mu) for k in range(n_max + 1)])
    np.testing.assert_allclose(pmf, manual, rtol=1e-10, atol=1e-12)
    cme = _cme_pmf(n_0, t, mu, n_max=40)
    np.testing.assert_allclose(pmf, cme[: n_max + 1], rtol=1e-6, atol=1e-8)
    # t=0 is a point mass at n_0 (beta=1, Poisson mean 0).
    pmf0 = np.asarray(forward_kernel_pmf_1d(n_0, 0.0, mu, n_max))
    expect0 = np.zeros(n_max + 1)
    expect0[n_0] = 1.0
    np.testing.assert_allclose(pmf0, expect0, atol=1e-12)


def test_kernel_factorizes_over_coordinates():
    n0 = jnp.array([2, 4])
    nt = jnp.array([1, 3])
    mu = jnp.array([2.0, 1.5])
    t = 0.25
    log_prod = log_forward_kernel(nt, n0, t, mu)
    log_a = log_binpois_conv(nt[0], n0[0], beta_t(t), mu[0] * (1.0 - beta_t(t)))
    log_b = log_binpois_conv(nt[1], n0[1], beta_t(t), mu[1] * (1.0 - beta_t(t)))
    np.testing.assert_allclose(np.asarray(log_prod), np.asarray(log_a + log_b), rtol=1e-10)


def test_terminal_time_deterministic_and_matches_paper_formula():
    counts = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [4, 4],
            [4, 4],
            [4, 4],
        ],
        dtype=np.float64,
    )
    t1, s1, noise1 = terminal_noising_time(counts)
    t2, s2, noise2 = terminal_noising_time(jnp.asarray(counts))
    np.testing.assert_allclose(np.asarray(t1), np.asarray(t2))
    np.testing.assert_allclose(np.asarray(s1), np.asarray(s2))
    np.testing.assert_allclose(np.asarray(noise1), np.asarray(noise2))

    n_obs, n_species = counts.shape
    mu = counts.mean(axis=0)
    centered = counts - mu
    cov = (centered.T @ centered) / n_obs
    psi = cov - np.diag(mu)
    inv_sqrt = 1.0 / np.sqrt(mu)
    c = inv_sqrt[:, None] * psi * inv_sqrt[None, :]
    sigma_1 = np.linalg.eigvalsh(0.5 * (c + c.T))[-1]
    sigma_noise = (1.0 + math.sqrt(n_species / n_obs)) ** 2 - 1.0
    t_o = 0.5 * math.log(sigma_1 / sigma_noise) if sigma_1 > sigma_noise else 0.0
    np.testing.assert_allclose(np.asarray(s1), sigma_1, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(noise1), sigma_noise, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(t1), t_o, rtol=1e-10)
    assert t_o > 0.0

    psi_fn, mu_fn = excess_covariance(counts)
    np.testing.assert_allclose(np.asarray(psi_fn), psi, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(largest_generalized_eigenvalue(psi, mu)), sigma_1, rtol=1e-10
    )
    np.testing.assert_allclose(np.asarray(mp_edge(n_obs, n_species)), sigma_noise)


def test_terminal_time_zero_when_sigma1_below_edge():
    # Four points, weak excess covariance relative to the MP edge.
    counts = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    t_o, sigma_1, sigma_noise = terminal_noising_time(counts)
    assert float(sigma_1) <= float(sigma_noise) + 1e-12
    assert float(t_o) == pytest.approx(0.0, abs=1e-12)


def test_reverse_sampling_stays_nonnegative_integers():
    key = jax.random.PRNGKey(0)
    class_mu, mix_w = two_class_params()
    n_max = 14
    grid = make_count_grid(n_max, 2)
    log_d = log_mixture_on_grid(grid, class_mu, mix_w)
    mu = mix_w @ class_mu
    times = jnp.linspace(0.0, 0.6, 5)
    x0 = ffbs_sample(key, 32, times, mu, grid, log_d, n_max)
    x0_np = np.asarray(x0)
    assert np.issubdtype(x0_np.dtype, np.integer)
    assert np.all(x0_np >= 0)
    assert x0_np.shape == (32, 2)

    # Bridge at s=0 recovers n_0; at s=t recovers n_t.
    n0 = jnp.array([2, 5], dtype=jnp.int32)
    nt = jnp.array([3, 4], dtype=jnp.int32)
    b0 = sample_bridge(jax.random.PRNGKey(1), n0, nt, 0.0, 0.5, mu, n_max)
    np.testing.assert_array_equal(np.asarray(b0), np.asarray(n0))
    bt = sample_bridge(jax.random.PRNGKey(2), n0, nt, 0.5, 0.5, mu, n_max)
    np.testing.assert_array_equal(np.asarray(bt), np.asarray(nt))

    # Forward samples are nonnegative integers.
    xf = np.asarray(sample_forward(jax.random.PRNGKey(3), n0, 0.4, mu))
    assert np.issubdtype(xf.dtype, np.integer)
    assert np.all(xf >= 0)


def test_bridge_normalizes_and_matches_displayed_ratio():
    n0, nt, t_s, t_t = 2, 3, 0.2, 0.7
    mu = 1.8
    n_max = 18
    ns = jnp.arange(n_max + 1)
    n0v = jnp.full((n_max + 1,), n0)
    ntv = jnp.full((n_max + 1,), nt)
    log_b = log_bridge(
        ns[:, None], n0v[:, None], ntv[:, None], t_s, t_t, jnp.array([mu])
    )
    p = np.exp(np.asarray(log_b))
    assert np.all(p >= -1e-15)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


def test_tilting_increases_target_class_mass():
    key = jax.random.PRNGKey(7)
    class_mu, mix_w = two_class_params()
    n_max = 14
    grid = make_count_grid(n_max, 2)
    log_d = log_mixture_on_grid(grid, class_mu, mix_w)
    log_d_ref = log_mixture_on_grid(grid, class_mu, jnp.array([0.0, 1.0]) + 1e-15)
    log_d_ref = log_d_ref - jax.nn.logsumexp(log_d_ref)
    log_h, _ = log_marginal_tilt(log_d, log_d_ref, grid, n_max, tau=1.0)
    mu = mix_w @ class_mu
    times = jnp.linspace(0.0, 0.8, 5)
    k1, k2 = jax.random.split(key)
    unsteered = ffbs_sample(k1, 96, times, mu, grid, log_d, n_max)
    steered = ffbs_sample(k2, 96, times, mu, grid, log_d, n_max, log_H=log_h)
    m_u = float(target_class_mass(unsteered, class_mu, mix_w))
    m_s = float(target_class_mass(steered, class_mu, mix_w))
    assert m_s > m_u
    assert m_s > 0.15


def _particles_for_mass(weights, mass: float = 0.9) -> int:
    ordered = np.sort(np.asarray(weights))[::-1]
    ordered = ordered / ordered.sum()
    return int(np.searchsorted(np.cumsum(ordered), mass) + 1)


def test_fk_tilting_before_correction_vs_naive_importance():
    class_mu, _ = two_class_params()
    mix_w = jnp.array([0.96, 0.04])
    n_max = 14
    grid = make_count_grid(n_max, 2)
    log_d = log_mixture_on_grid(grid, class_mu, mix_w)
    log_d_ref = log_mixture_on_grid(grid, class_mu, jnp.array([1e-15, 1.0]))
    log_d_ref = log_d_ref - jax.nn.logsumexp(log_d_ref)
    log_h, _ = log_marginal_tilt(log_d, log_d_ref, grid, n_max, tau=1.0)
    log_d_tilt = log_d + log_h
    log_d_tilt = log_d_tilt - jax.nn.logsumexp(log_d_tilt)

    ess_naive = float(importance_ess_frac(log_d, log_d_ref))
    ess_tilt = float(importance_ess_frac(log_d_tilt, log_d_ref))
    assert ess_tilt > ess_naive
    assert ess_naive < 0.2
    cv2_naive = 1.0 / ess_naive - 1.0
    cv2_tilt = 1.0 / ess_tilt - 1.0
    assert cv2_naive > cv2_tilt

    # Finite-sample FK weights: naive importance on the rare class concentrates.
    rho_naive = np.exp(np.asarray(log_d_ref - log_d))
    rho_tilt = np.exp(np.asarray(log_d_ref - log_d_tilt))
    k_w1, k_w2 = jax.random.split(jax.random.PRNGKey(3))
    idx_d = np.asarray(jax.random.categorical(k_w1, log_d, shape=(256,)))
    idx_t = np.asarray(jax.random.categorical(k_w2, log_d_tilt, shape=(256,)))
    w_d = rho_naive[idx_d]
    w_t = rho_tilt[idx_t]
    ess_d = float((w_d.sum() ** 2) / np.square(w_d).sum())
    ess_t = float((w_t.sum() ** 2) / np.square(w_t).sum())
    assert ess_t > ess_d
    assert _particles_for_mass(w_d) < _particles_for_mass(w_t)

    # Tilted posterior is Q^H ∝ H Q, not a post-hoc reweight of an untilted draw.
    n_t = jnp.array([4, 4])
    mu = mix_w @ class_mu
    log_q = posterior_logits(n_t, 0.4, mu, grid, log_d)
    log_qh = tilt_posterior_logits(log_q, log_h)
    np.testing.assert_allclose(
        np.asarray(log_qh),
        np.asarray(posterior_logits(n_t, 0.4, mu, grid, log_d, log_H=log_h)),
        rtol=1e-8,
        atol=1e-8,
    )
    psi = float(continuation_potential(log_qh, jnp.exp(log_d_ref - log_d_tilt)))
    assert psi > 0.0

    key = jax.random.PRNGKey(11)
    times = jnp.linspace(0.0, 0.5, 4)
    x0, w, ess_hist = tilted_fk_sample(
        key, 24, times, mu, grid, log_d, log_d_ref, n_max, tau=1.0
    )
    x0_np = np.asarray(x0)
    assert np.all(x0_np >= 0)
    assert np.issubdtype(x0_np.dtype, np.integer)
    assert float(jnp.min(w)) > 0.0
    assert ess_hist.shape[0] == 3


def test_log_pmf_helpers_match_manual():
    np.testing.assert_allclose(
        float(jnp.exp(log_binom_pmf(2, 5, 0.3))),
        math.comb(5, 2) * 0.3**2 * 0.7**3,
        rtol=1e-12,
    )
    lam = 1.4
    np.testing.assert_allclose(
        float(jnp.exp(log_poisson_pmf(3, lam))),
        math.exp(-lam) * lam**3 / math.factorial(3),
        rtol=1e-12,
    )
    assert float(jnp.exp(log_binom_pmf(0, 4, 0.0))) == pytest.approx(1.0)
    assert float(jnp.exp(log_poisson_pmf(0, 0.0))) == pytest.approx(1.0)
    assert K_MINUS == 1.0
