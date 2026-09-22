#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from statevc import KNNVCBackend, StateVCConfig, convert_features


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="StateVC one-shot training-free voice conversion")
    p.add_argument("--source", required=True, help="source wav")
    p.add_argument("--reference", required=True, help="target-speaker reference wav")
    p.add_argument("--output", required=True, help="output wav")
    p.add_argument("--device", default="cuda")
    p.add_argument("--operator", choices=["mean", "diagonal", "block", "none"], default="mean")
    p.add_argument("--routing", choices=["soft", "hard"], default="soft")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--cluster-dim", type=int, default=24)
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument("--n-min", type=int, default=20)
    p.add_argument("--m-min", type=float, default=20.0)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--block-size", type=int, default=2)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--diagnostics", default=None, help="optional JSON diagnostics path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = StateVCConfig(
        cluster_dim=args.cluster_dim,
        k_max=args.k_max,
        n_min=args.n_min,
        m_min=args.m_min,
        smooth_window=args.smooth_window,
        strength=args.strength,
        operator=args.operator,
        routing=args.routing,
        block_size=args.block_size,
        eps=args.eps,
    )

    backend = KNNVCBackend(device=args.device, prematched=True)
    with torch.inference_mode():
        h_s = backend.extract(args.source)
        h_r = backend.extract(args.reference)
        result = convert_features(h_s, h_r, cfg)
        wav = backend.vocode(result.features)
    backend.save(args.output, wav)

    diag_path = Path(args.diagnostics) if args.diagnostics else Path(args.output).with_suffix(".json")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(args.source),
        "reference": str(args.reference),
        "output": str(args.output),
        "config": cfg.__dict__,
        "statevc": result.diagnostics,
    }
    diag_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
