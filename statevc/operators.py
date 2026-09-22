from __future__ import annotations

from typing import Tuple

import torch

from .config import StateVCConfig


def _weighted_mean(x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mass = w.sum()
    mean = (w[:, None] * x).sum(dim=0) / mass.clamp_min(torch.finfo(x.dtype).eps)
    return mean, mass


def _weighted_var(x: torch.Tensor, w: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    mass = w.sum().clamp_min(torch.finfo(x.dtype).eps)
    return (w[:, None] * (x - mean).square()).sum(dim=0) / mass


def _weighted_cov(x: torch.Tensor, w: torch.Tensor, mean: torch.Tensor, eps: float) -> torch.Tensor:
    mass = w.sum().clamp_min(torch.finfo(x.dtype).eps)
    xc = x - mean
    cov = (xc * w[:, None]).transpose(0, 1) @ xc / mass
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return 0.5 * (cov + cov.transpose(0, 1)) + eps * eye


def _spd_pow(a: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    # float64 eigendecomposition is materially more stable for 2-D covariance blocks.
    original_dtype = a.dtype
    a64 = a.double()
    evals, evecs = torch.linalg.eigh(0.5 * (a64 + a64.transpose(-1, -2)))
    evals = evals.clamp_min(eps).pow(power)
    out = (evecs * evals.unsqueeze(0)) @ evecs.transpose(-1, -2)
    return out.to(original_dtype)


def gaussian_transport_map(cov_s: torch.Tensor, cov_r: torch.Tensor, eps: float) -> torch.Tensor:
    """Bures / Gaussian 2-Wasserstein linear covariance transport map."""
    s_half = _spd_pow(cov_s, 0.5, eps)
    s_inv_half = _spd_pow(cov_s, -0.5, eps)
    middle = s_half @ cov_r @ s_half
    middle_half = _spd_pow(middle, 0.5, eps)
    return s_inv_half @ middle_half @ s_inv_half


def _routing_weights(gamma_s: torch.Tensor, routing: str) -> torch.Tensor:
    if routing == "soft":
        return gamma_s
    labels = gamma_s.argmax(dim=1)
    return torch.nn.functional.one_hot(labels, num_classes=gamma_s.shape[1]).to(gamma_s.dtype)


def apply_mean(
    h_s: torch.Tensor,
    h_r: torch.Tensor,
    gamma_s: torch.Tensor,
    gamma_r: torch.Tensor,
    cfg: StateVCConfig,
) -> torch.Tensor:
    """Final StateVC operator: posterior-weighted local mean translations."""
    k_eff = gamma_s.shape[1]
    global_delta = h_r.mean(dim=0) - h_s.mean(dim=0)
    deltas = []
    for k in range(k_eff):
        mu_s, mass_s = _weighted_mean(h_s, gamma_s[:, k])
        mu_r, mass_r = _weighted_mean(h_r, gamma_r[:, k])
        if mass_s.item() >= cfg.m_min and mass_r.item() >= cfg.m_min:
            deltas.append(mu_r - mu_s)
        else:
            deltas.append(global_delta)
    delta = torch.stack(deltas, dim=0)
    w = _routing_weights(gamma_s, cfg.routing)
    update = w @ delta
    return h_s + cfg.strength * update


def apply_diagonal(
    h_s: torch.Tensor,
    h_r: torch.Tensor,
    gamma_s: torch.Tensor,
    gamma_r: torch.Tensor,
    cfg: StateVCConfig,
) -> torch.Tensor:
    """Coordinate-wise state transport used by the Diagonal ablation."""
    k_eff = gamma_s.shape[1]
    w_route = _routing_weights(gamma_s, cfg.routing)
    global_delta = h_r.mean(dim=0) - h_s.mean(dim=0)
    transformed = []

    for k in range(k_eff):
        mu_s, mass_s = _weighted_mean(h_s, gamma_s[:, k])
        mu_r, mass_r = _weighted_mean(h_r, gamma_r[:, k])
        if mass_s.item() >= cfg.m_min and mass_r.item() >= cfg.m_min:
            var_s = _weighted_var(h_s, gamma_s[:, k], mu_s)
            var_r = _weighted_var(h_r, gamma_r[:, k], mu_r)
            scale = torch.sqrt((var_r + cfg.eps) / (var_s + cfg.eps))
            y = (h_s - mu_s) * scale + mu_r
        else:
            # Same low-support policy as the final Mean formulation: use only
            # the utterance-level mean displacement rather than estimate an
            # unreliable local scale.
            y = h_s + global_delta
        transformed.append(y)

    y_all = torch.stack(transformed, dim=1)  # [T,K,D]
    mixed = (w_route[:, :, None] * y_all).sum(dim=1)
    return (1.0 - cfg.strength) * h_s + cfg.strength * mixed


def apply_block(
    h_s: torch.Tensor,
    h_r: torch.Tensor,
    gamma_s: torch.Tensor,
    gamma_r: torch.Tensor,
    permutation: torch.Tensor,
    cfg: StateVCConfig,
) -> torch.Tensor:
    """Block-wise Gaussian/MKL covariance transport ablation.

    The full 1024-D WavLM space is first reordered by source temporal variance,
    then split into contiguous blocks of size `block_size` (2 in the paper).
    Each valid state receives an independent Gaussian covariance map per block.
    """
    k_eff = gamma_s.shape[1]
    w_route = _routing_weights(gamma_s, cfg.routing)

    x_s = h_s[:, permutation]
    x_r = h_r[:, permutation]
    global_delta = x_r.mean(dim=0) - x_s.mean(dim=0)

    state_outputs = []
    d = x_s.shape[1]
    for k in range(k_eff):
        mu_s, mass_s = _weighted_mean(x_s, gamma_s[:, k])
        mu_r, mass_r = _weighted_mean(x_r, gamma_r[:, k])
        if mass_s.item() < cfg.m_min or mass_r.item() < cfg.m_min:
            state_outputs.append(x_s + global_delta)
            continue

        blocks = []
        for start in range(0, d, cfg.block_size):
            end = min(start + cfg.block_size, d)
            xs_b = x_s[:, start:end]
            xr_b = x_r[:, start:end]
            mus_b = mu_s[start:end]
            mur_b = mu_r[start:end]
            cov_s = _weighted_cov(xs_b, gamma_s[:, k], mus_b, cfg.eps)
            cov_r = _weighted_cov(xr_b, gamma_r[:, k], mur_b, cfg.eps)
            a = gaussian_transport_map(cov_s, cov_r, cfg.eps)
            # Row-vector convention: x' = mu_r + (x-mu_s) A.
            y_b = mus_b.new_tensor(0.0) + (xs_b - mus_b) @ a + mur_b
            blocks.append(y_b)
        state_outputs.append(torch.cat(blocks, dim=1))

    y_ranked = torch.stack(state_outputs, dim=1)  # [T,K,D]
    mixed_ranked = (w_route[:, :, None] * y_ranked).sum(dim=1)

    # Undo the source-variance permutation.
    mixed = torch.empty_like(mixed_ranked)
    mixed[:, permutation] = mixed_ranked
    return (1.0 - cfg.strength) * h_s + cfg.strength * mixed


def apply_operator(
    h_s: torch.Tensor,
    h_r: torch.Tensor,
    gamma_s: torch.Tensor,
    gamma_r: torch.Tensor,
    permutation: torch.Tensor,
    cfg: StateVCConfig,
) -> torch.Tensor:
    if cfg.operator == "none":
        return h_s.clone()
    if cfg.operator == "mean":
        return apply_mean(h_s, h_r, gamma_s, gamma_r, cfg)
    if cfg.operator == "diagonal":
        return apply_diagonal(h_s, h_r, gamma_s, gamma_r, cfg)
    if cfg.operator == "block":
        return apply_block(h_s, h_r, gamma_s, gamma_r, permutation, cfg)
    raise ValueError(f"unknown operator: {cfg.operator}")
