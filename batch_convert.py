#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from statevc import KNNVCBackend, StateVCConfig, convert_features


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch StateVC conversion from a TSV manifest")
    p.add_argument("--manifest", required=True, help="TSV with source, reference, output columns")
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
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
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

    with open(args.manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"source", "reference", "output"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"manifest needs columns: {sorted(required)}")
        rows = list(reader)

    for i, row in enumerate(rows, 1):
        out = Path(row["output"])
        if out.exists() and not args.overwrite:
            print(f"[{i}/{len(rows)}] skip existing: {out}")
            continue
        print(f"[{i}/{len(rows)}] {row['source']} -> {row['reference']} -> {out}")
        with torch.inference_mode():
            h_s = backend.extract(row["source"])
            h_r = backend.extract(row["reference"])
            result = convert_features(h_s, h_r, cfg)
            wav = backend.vocode(result.features)
        backend.save(out, wav)
        out.with_suffix(".json").write_text(
            json.dumps(
                {
                    "source": row["source"],
                    "reference": row["reference"],
                    "output": row["output"],
                    "config": cfg.__dict__,
                    "statevc": result.diagnostics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
