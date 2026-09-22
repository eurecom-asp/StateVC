from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from .config import StateVCConfig


@dataclass
class GMMSelection:
    model: GaussianMixture
    k_eff: int
    bic: float
    support: np.ndarray
    valid_candidates: List[Tuple[int, float, np.ndarray]]


def fit_shared_gmm(z_s: np.ndarray, z_r: np.ndarray, cfg: StateVCConfig) -> GMMSelection:
    """Fit the pair-specific shared diagonal GMM and select K by BIC.

    Source and reference frames are pooled without frame alignment or
    equal-duration resampling. Candidate K values are 1..k_max. A candidate is
    eligible only when every component receives at least n_min hard-assigned
    pooled frames. Among eligible candidates, the one with minimum BIC is used.
    """
    cfg.validate()
    if z_s.ndim != 2 or z_r.ndim != 2:
        raise ValueError("z_s and z_r must be 2-D arrays")
    if z_s.shape[1] != z_r.shape[1]:
        raise ValueError("z_s and z_r must have the same feature dimension")

    z = np.concatenate([z_s, z_r], axis=0).astype(np.float64, copy=False)
    if z.shape[0] == 0:
        raise ValueError("cannot fit a GMM on zero frames")

    valid = []
    models = {}
    max_k = min(cfg.k_max, z.shape[0])
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=cfg.gmm_seed,
                n_init=cfg.gmm_n_init,
                max_iter=cfg.gmm_max_iter,
                reg_covar=cfg.gmm_reg_covar,
                init_params=cfg.gmm_init_params,
            )
            gmm.fit(z)
            posterior = gmm.predict_proba(z)
            labels = posterior.argmax(axis=1)
            support = np.bincount(labels, minlength=k)
            models[k] = (gmm, support)
            if np.all(support >= cfg.n_min):
                bic = float(gmm.bic(z))
                valid.append((k, bic, support.copy()))
        except (ValueError, FloatingPointError):
            continue

    if valid:
        k_eff, bic, support = min(valid, key=lambda item: item[1])
        gmm = models[k_eff][0]
        return GMMSelection(
            model=gmm,
            k_eff=k_eff,
            bic=bic,
            support=support,
            valid_candidates=valid,
        )

    # Very short pairs can make the n_min support constraint impossible. The
    # final paper setup normally avoids this case. For a usable public API we
    # fall back to K=1 rather than failing conversion outright.
    if 1 not in models:
        gmm = GaussianMixture(
            n_components=1,
            covariance_type="diag",
            random_state=cfg.gmm_seed,
            n_init=cfg.gmm_n_init,
            max_iter=cfg.gmm_max_iter,
            reg_covar=cfg.gmm_reg_covar,
            init_params=cfg.gmm_init_params,
        ).fit(z)
        support = np.array([z.shape[0]], dtype=np.int64)
    else:
        gmm, support = models[1]
    return GMMSelection(
        model=gmm,
        k_eff=1,
        bic=float(gmm.bic(z)),
        support=support,
        valid_candidates=[],
    )
