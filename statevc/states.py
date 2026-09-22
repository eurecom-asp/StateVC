from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import StateVCConfig
from .gmm import GMMSelection, fit_shared_gmm


@dataclass
class StateAssignment:
    permutation: torch.Tensor
    routing_indices: torch.Tensor
    gamma_s: torch.Tensor
    gamma_r: torch.Tensor
    selection: GMMSelection


def source_variance_order(h_s: torch.Tensor) -> torch.Tensor:
    """Rank full WavLM coordinates by temporal std of the source utterance."""
    if h_s.ndim != 2:
        raise ValueError("h_s must have shape [T, D]")
    return torch.argsort(h_s.std(dim=0), descending=True)


def smooth_posteriors(gamma: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Edge-replicated moving-average smoothing followed by row renormalization."""
    if gamma.ndim != 2:
        raise ValueError("gamma must have shape [T, K]")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if window == 1 or gamma.shape[0] == 0:
        denom = gamma.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(gamma.dtype).eps)
        return gamma / denom

    j = window // 2
    # [T,K] -> [1,K,T], replicate temporal edges exactly.
    x = gamma.transpose(0, 1).unsqueeze(0)
    x = torch.nn.functional.pad(x, (j, j), mode="replicate")
    kernel = torch.ones((gamma.shape[1], 1, window), device=gamma.device, dtype=gamma.dtype) / window
    y = torch.nn.functional.conv1d(x, kernel, groups=gamma.shape[1])
    y = y.squeeze(0).transpose(0, 1)
    return y / y.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(y.dtype).eps)


def infer_shared_states(h_s: torch.Tensor, h_r: torch.Tensor, cfg: StateVCConfig) -> StateAssignment:
    """Construct pair-specific shared states from source-derived routing coordinates."""
    cfg.validate()
    if h_s.ndim != 2 or h_r.ndim != 2:
        raise ValueError("h_s and h_r must both have shape [T, D]")
    if h_s.shape[1] != h_r.shape[1]:
        raise ValueError("h_s and h_r must have the same WavLM dimension")

    perm = source_variance_order(h_s)
    d_c = min(cfg.cluster_dim, h_s.shape[1])
    route_idx = perm[:d_c]

    z_s = h_s[:, route_idx].detach().cpu().double().numpy()
    z_r = h_r[:, route_idx].detach().cpu().double().numpy()
    selection = fit_shared_gmm(z_s, z_r, cfg)

    p_s = selection.model.predict_proba(z_s)
    p_r = selection.model.predict_proba(z_r)
    gamma_s = torch.as_tensor(p_s, dtype=h_s.dtype, device=h_s.device)
    gamma_r = torch.as_tensor(p_r, dtype=h_r.dtype, device=h_r.device)

    gamma_s = smooth_posteriors(gamma_s, cfg.smooth_window)
    gamma_r = smooth_posteriors(gamma_r, cfg.smooth_window)

    return StateAssignment(
        permutation=perm,
        routing_indices=route_idx,
        gamma_s=gamma_s,
        gamma_r=gamma_r,
        selection=selection,
    )
