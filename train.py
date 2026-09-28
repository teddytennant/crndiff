"""Tiny CPU/GPU smoke: birth-death FFBS and tilted FK on a two-class count toy."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crndiff.ffbs import ffbs_sample
from crndiff.steer import importance_ess_frac, log_marginal_tilt, tilted_fk_sample
from crndiff.terminal import terminal_noising_time
from crndiff.toy import (
    log_mixture_on_grid,
    make_count_grid,
    target_class_mass,
    two_class_params,
    two_class_samples,
)


def main():
    key = jax.random.PRNGKey(0)
    class_mu, mix_w = two_class_params()
    key, k_data = jax.random.split(key)
    counts, _ = two_class_samples(k_data, 64, class_mu, mix_w)
    t_o, sigma_1, sigma_noise = terminal_noising_time(counts)
    t_o = float(t_o)
    t_use = max(t_o, 0.35)

    n_max = 14
    grid = make_count_grid(n_max, 2)
    log_d = log_mixture_on_grid(grid, class_mu, mix_w)
    log_d_ref = log_mixture_on_grid(grid, class_mu, jnp.array([1e-15, 1.0]))
    log_d_ref = log_d_ref - jax.nn.logsumexp(log_d_ref)
    log_h, _ = log_marginal_tilt(log_d, log_d_ref, grid, n_max, tau=1.0)
    mu = mix_w @ class_mu
    times = jnp.linspace(0.0, t_use, 5)

    key, k_u, k_s, k_fk = jax.random.split(key, 4)
    unsteered = ffbs_sample(k_u, 48, times, mu, grid, log_d, n_max)
    steered = ffbs_sample(k_s, 48, times, mu, grid, log_d, n_max, log_H=log_h)
    fk_x0, fk_w, ess_hist = tilted_fk_sample(
        k_fk, 32, times, mu, grid, log_d, log_d_ref, n_max, tau=1.0
    )

    log_d_tilt = log_d + log_h
    log_d_tilt = log_d_tilt - jax.nn.logsumexp(log_d_tilt)
    payload = {
        "T_O": t_o,
        "T_used": t_use,
        "sigma_1": float(sigma_1),
        "sigma_noise": float(sigma_noise),
        "n_obs": int(counts.shape[0]),
        "n_species": int(counts.shape[1]),
        "unconditional_mean": np.asarray(unsteered.mean(axis=0), dtype=float).tolist(),
        "steered_mean": np.asarray(steered.mean(axis=0), dtype=float).tolist(),
        "fk_mean": np.asarray(fk_x0.mean(axis=0), dtype=float).tolist(),
        "unconditional_target_mass": float(target_class_mass(unsteered, class_mu, mix_w)),
        "steered_target_mass": float(target_class_mass(steered, class_mu, mix_w)),
        "fk_target_mass": float(target_class_mass(fk_x0, class_mu, mix_w)),
        "naive_ess_frac": float(importance_ess_frac(log_d, log_d_ref)),
        "tilted_ess_frac": float(importance_ess_frac(log_d_tilt, log_d_ref)),
        "fk_final_ess": float(ess_hist[-1]),
        "fk_weight_min": float(jnp.min(fk_w)),
        "device": jax.default_backend(),
    }
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
