# StateVC

**StateVC: Shared-State Local Translations for Training-Free Voice Conversion**

This repository contains the ICASSP implementation of StateVC, a training-free one-shot voice conversion method that defines a shared pair-specific acoustic state space before applying local source-to-reference transformations.

## Final ICASSP system

The released default reproduces the final StateVC configuration:

- frozen **WavLM-Large layer 6**, feature dimension `D=1024`;
- rank WavLM coordinates by **source temporal standard deviation**;
- use the source-derived top `d_c=24` coordinates for both source and reference state routing;
- fit one **pair-specific shared diagonal GMM** to all pooled, unaligned source and reference frames;
- candidate state counts `K in {1,2,3,4}`;
- discard a candidate GMM if any component has fewer than `n_min=20` hard-assigned pooled frames;
- select `K_eff` by minimum BIC among valid candidates;
- obtain source/reference GMM posteriors, apply a 5-frame edge-replicated moving average, and renormalize across states;
- compute state statistics in the **full 1024-D WavLM space**;
- require posterior mass `m_min=20` in both utterances for a local state update; otherwise use the utterance-level reference-minus-source mean displacement;
- apply the **Mean** operator with **soft routing** and conversion strength `lambda=1`;
- synthesize the edited WavLM sequence with the frozen **prematched WavLM-conditioned HiFi-GAN released with kNN-VC**.

There is **no frame alignment** between source and reference. The shared GMM is fitted to the direct concatenation of all source and reference routing frames; therefore, a longer utterance contributes more observations to mixture estimation.

## Method

Let `H_s in R^(T_s x 1024)` and `H_r in R^(T_r x 1024)` be WavLM-Large layer-6 features.
The source-derived variance ranking is used to select the same 24 routing coordinates from both utterances, yielding `Z_s` and `Z_r`.
A diagonal GMM is fitted to `[Z_s; Z_r]`, and its smoothed posteriors are denoted by `Gamma_s` and `Gamma_r`.

For state `k`, the full-space posterior masses and means are

```text
M_q,k  = sum_t Gamma_q[t,k]
mu_q,k = sum_t Gamma_q[t,k] H_q[t] / M_q,k
```

The final Mean operator uses

```text
delta_k = mu_r,k - mu_s,k
```

when both `M_s,k >= 20` and `M_r,k >= 20`; otherwise

```text
delta_k = mean(H_r) - mean(H_s).
```

With soft routing,

```text
H'_s[t] = H_s[t] + lambda * sum_k Gamma_s[t,k] delta_k,
```

and the final configuration uses `lambda=1`.

## Operator ablations

The repository also includes the operators used in the paper analyses:

- **Mean**: state-specific full-space mean translation. This is the final StateVC system.
- **Diagonal**: coordinate-wise state transport,
  `mu_r + (x-mu_s) * sqrt((var_r+eps)/(var_s+eps))`.
- **Block**: reorder all 1024 WavLM dimensions by source temporal variance and split them into contiguous 2-D blocks. Within each state and block, apply the Gaussian/Bures (MKL) covariance map
  `A = S_s^(-1/2) (S_s^(1/2) S_r S_s^(1/2))^(1/2) S_s^(-1/2)`.
  The routing dimension `d_c=24` is independent of the transport block size `2`.

The main Mean operator does not require covariance regularization. `eps=1e-4` is used only for the Diagonal/Block numerical stabilization in this release.

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/eurecom-asp/StateVC.git
cd StateVC
pip install -e .
```

The first inference call uses `torch.hub` to download the official `bshall/knn-vc` code pinned to commit `c616845c4e309e24d5927f15adbdf277a3d65358`, together with the released WavLM-Large and prematched HiFi-GAN checkpoints.

## Single-pair conversion

Final paper configuration:

```bash
python convert.py \
  --source /path/to/source.wav \
  --reference /path/to/reference.wav \
  --output outputs/statevc.wav \
  --operator mean \
  --routing soft \
  --cluster-dim 24 \
  --k-max 4 \
  --n-min 20 \
  --m-min 20 \
  --smooth-window 5 \
  --strength 1.0
```

or simply:

```bash
bash scripts/run_final_mean.sh source.wav reference.wav outputs/statevc.wav
```

A JSON file is written next to the output waveform with `K_eff`, BIC candidates, hard component support, selected routing dimensions, and source/reference posterior masses.

## Batch conversion

Prepare a tab-separated manifest:

```text
source  reference  output
/path/src1.wav  /path/ref1.wav  outputs/001.wav
/path/src2.wav  /path/ref2.wav  outputs/002.wav
```

Then run:

```bash
python batch_convert.py --manifest pairs.tsv --device cuda
```

## Reproduce the operator ablation for one pair

```bash
bash scripts/run_operator_ablation.sh source.wav reference.wav outputs/operators
```

This produces `mean.wav`, `diagonal.wav`, and `block.wav` using the same state construction and soft routing.

## Important implementation distinctions

`n_min` and `m_min` are deliberately different quantities:

1. `n_min=20` is an **integer hard-assigned pooled-frame count** used only to decide whether a candidate GMM is eligible for BIC selection.
2. `m_min=20` is a **per-utterance posterior mass** used to decide whether a state-specific local transformation has sufficient source and reference support.

The top-24 routing dimensions are not a transport subspace. State means and the final Mean update operate on the complete 1024-D WavLM representation.

## Tests

Core algorithm tests do not download WavLM or HiFi-GAN:

```bash
python -m pytest -q
```

## Backend

StateVC uses kNN-VC only as a frozen feature/synthesis backend: its WavLM-Large layer-6 extractor and prematched WavLM-conditioned HiFi-GAN are loaded through `torch.hub`. StateVC does not perform k-nearest-neighbor matching.

## Repository scope

This release intentionally contains the final ICASSP StateVC method and the directly associated operator/routing ablations only. Earlier GeoTF/SLOT variants, optimal-transport retrieval systems, accent-conversion pipelines, and progressive speech-editor experiments are not part of this repository.
