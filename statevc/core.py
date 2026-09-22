from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch

from .config import StateVCConfig
from .operators import apply_operator
from .states import infer_shared_states


@dataclass
class ConversionResult:
    features: torch.Tensor
    diagnostics: Dict[str, Any]


@torch.inference_mode()
def convert_features(h_s: torch.Tensor, h_r: torch.Tensor, cfg: StateVCConfig) -> ConversionResult:
    """Apply StateVC to source/reference WavLM feature sequences.

    Args:
        h_s: [T_s, D] source WavLM-Large layer-6 features.
        h_r: [T_r, D] reference WavLM-Large layer-6 features.
        cfg: StateVC configuration.
    """
    cfg.validate()
    if h_s.ndim != 2 or h_r.ndim != 2:
        raise ValueError("h_s and h_r must both have shape [T, D]")
    if h_s.shape[1] != h_r.shape[1]:
        raise ValueError("source/reference dimensions differ")
    if h_s.device != h_r.device:
        h_r = h_r.to(h_s.device)
    if h_s.dtype != h_r.dtype:
        h_r = h_r.to(h_s.dtype)

    assignment = infer_shared_states(h_s, h_r, cfg)
    out = apply_operator(
        h_s,
        h_r,
        assignment.gamma_s,
        assignment.gamma_r,
        assignment.permutation,
        cfg,
    )

    masses_s = assignment.gamma_s.sum(dim=0).detach().cpu().tolist()
    masses_r = assignment.gamma_r.sum(dim=0).detach().cpu().tolist()
    diag = {
        "k_eff": assignment.selection.k_eff,
        "bic": assignment.selection.bic,
        "hard_support": assignment.selection.support.tolist(),
        "valid_bic_candidates": [
            {"k": k, "bic": bic, "hard_support": support.tolist()}
            for k, bic, support in assignment.selection.valid_candidates
        ],
        "routing_indices": assignment.routing_indices.detach().cpu().tolist(),
        "posterior_mass_source": masses_s,
        "posterior_mass_reference": masses_r,
        "operator": cfg.operator,
        "routing": cfg.routing,
        "strength": cfg.strength,
    }
    return ConversionResult(features=out, diagnostics=diag)
