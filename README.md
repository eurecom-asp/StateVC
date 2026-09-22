<div align="center">

# StateVC

**Shared-State Local Translations for Training-Free Voice Conversion**


[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-Voice%20Conversion-6C5CE7)](#)
[![Training](https://img.shields.io/badge/Training-Free-00A86B)](#)

</div>

---

## Overview

**StateVC** is a training-free one-shot voice conversion method that operates directly on frozen WavLM representations.

Given a source utterance and a target-speaker reference utterance, StateVC constructs a pair-specific shared acoustic-state space, estimates state-wise source-to-reference feature translations, and synthesizes the converted representation with a frozen WavLM-conditioned HiFi-GAN backend.

The method does not train or fine-tune a voice conversion model.

### Key properties

- **Training-free:** no additional VC model training or speaker adaptation.
- **One-shot:** one source utterance and one target reference utterance are sufficient.
- **Shared-state modeling:** source and reference frames are routed through the same pair-specific acoustic states.
- **Local conversion:** feature updates are estimated independently for different acoustic states.
- **Soft routing:** each source frame can combine multiple state-specific translations.
- **Frozen backend:** WavLM-Large and HiFi-GAN remain unchanged during conversion.

---

## Method at a Glance

StateVC consists of four stages.

### 1. Frozen speech representation

Source and reference waveforms are encoded with **WavLM-Large layer 6**, producing 1024-dimensional frame-level representations.

### 2. Pair-specific shared-state discovery

Feature coordinates are ranked by the temporal variance of the source utterance.

The top 24 coordinates are used as the routing representation for both source and reference.

A shared diagonal-covariance GMM is fitted to the pooled source and reference routing frames, with the number of states selected from 1 to 4 using BIC.

There is no source-reference frame alignment and no equal-duration resampling before GMM fitting.

### 3. Local feature translation

State statistics are estimated in the **full 1024-dimensional WavLM space**, not only in the 24-dimensional routing space.

The default system computes a source-to-reference mean displacement for each shared state and combines these local translations using smoothed source-state posteriors.

States with insufficient support fall back to an utterance-level source-to-reference mean displacement.

### 4. Waveform synthesis

The edited WavLM representation is synthesized directly with the frozen prematched WavLM-conditioned HiFi-GAN released with kNN-VC.

StateVC does **not** perform k-nearest-neighbor matching.

---

## Default Configuration

| Component | Setting |
| --- | --- |
| Encoder | WavLM-Large |
| WavLM layer | 6 |
| Feature dimension | 1024 |
| Routing dimensions | 24 |
| Routing-coordinate selection | Source temporal variance |
| State model | Shared diagonal GMM |
| Candidate state counts | 1, 2, 3, 4 |
| State-count selection | BIC |
| Minimum pooled hard support | 20 frames |
| Posterior smoothing | 5-frame moving average |
| Minimum source/reference posterior mass | 20 |
| Default transformation | Mean translation |
| Routing | Soft |
| Conversion strength | 1.0 |
| Decoder | Frozen prematched kNN-VC HiFi-GAN |

> **Implementation distinction:** the 24 selected dimensions are used only for state construction and routing.  
> State statistics and the default transformation are computed in the full 1024-dimensional WavLM space.

---

## Installation

```bash
git clone git@github.com:eurecom-asp/StateVC.git
cd StateVC
pip install -e .
```

For development and testing:

```bash
pip install pytest
python -m pytest -q
```

On the first inference run, the backend is loaded through `torch.hub`.

The kNN-VC source is pinned to:

```text
c616845c4e309e24d5927f15adbdf277a3d65358
```

---

## Quick Start

### Single-pair conversion

```bash
python convert.py     --source /path/to/source.wav     --reference /path/to/reference.wav     --output outputs/statevc.wav     --device cuda
```

The default command uses the released StateVC configuration:

```text
operator       = mean
routing        = soft
cluster_dim    = 24
k_max          = 4
n_min          = 20
m_min          = 20
smooth_window  = 5
strength       = 1.0
```

Equivalent convenience script:

```bash
bash scripts/run_final_mean.sh     /path/to/source.wav     /path/to/reference.wav     outputs/statevc.wav
```

---

## Diagnostics

For an output waveform such as

```text
outputs/statevc.wav
```

StateVC also writes

```text
outputs/statevc.json
```

with conversion diagnostics including:

- selected number of states;
- candidate BIC values;
- hard component support;
- selected routing dimensions;
- source posterior masses;
- reference posterior masses;
- transformation operator;
- routing mode;
- conversion strength.

These diagnostics are useful for checking the behavior of individual source-reference pairs.

---

## Batch Conversion

Create a tab-separated manifest:

```text
source	reference	output
/path/src1.wav	/path/ref1.wav	outputs/001.wav
/path/src2.wav	/path/ref2.wav	outputs/002.wav
```

Run:

```bash
python batch_convert.py     --manifest pairs.tsv     --device cuda
```

---

## Transformation Operators

The repository includes the default StateVC operator and two analysis operators.

### Mean

The default StateVC transformation.

Each shared state applies a full-space source-to-reference mean translation.

### Diagonal

A coordinate-wise Gaussian transformation based on state-specific source and reference means and variances.

### Block

A block-wise Gaussian covariance transport operator.

The full WavLM representation is reordered according to source temporal variance and partitioned into contiguous 2-dimensional blocks before the state-specific transformation is applied.

The **Diagonal** and **Block** operators are provided for analysis; the released default system uses **Mean**.

Run all operators on the same source-reference pair with:

```bash
bash scripts/run_operator_ablation.sh     source.wav     reference.wav     outputs/operators
```

Expected outputs:

```text
outputs/operators/
├── mean.wav
├── mean.json
├── diagonal.wav
├── diagonal.json
├── block.wav
└── block.json
```

---

## Repository Layout

```text
StateVC/
├── statevc/
│   ├── backend.py
│   ├── config.py
│   ├── core.py
│   ├── gmm.py
│   ├── operators.py
│   └── states.py
├── scripts/
│   ├── run_final_mean.sh
│   └── run_operator_ablation.sh
├── tests/
│   └── test_statevc.py
├── convert.py
├── batch_convert.py
├── example_manifest.tsv
├── REPRODUCIBILITY.md
├── requirements.txt
└── pyproject.toml
```

---

## Reproducibility

The implementation fixes the main state-construction settings used by the released system.

### Shared GMM

```text
covariance_type = diag
random_state    = 0
n_init          = 3
max_iter        = 100
reg_covar       = 1e-4
init_params     = kmeans
```

A candidate GMM is eligible for BIC selection only if every component receives at least 20 hard-assigned pooled frames.

For local state-wise conversion, a state-specific update is used only when both the source and the reference have posterior mass of at least 20 for that state.

Otherwise, StateVC uses the utterance-level mean displacement as the fallback transformation.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the implementation lock and additional details.

---

## Testing

Core algorithm tests can be run without downloading WavLM or HiFi-GAN:

```bash
python -m pytest -q
```

The test suite covers:

- source-derived variance ranking;
- posterior smoothing and normalization;
- shared-GMM support filtering;
- mean-translation identity behavior;
- global-shift recovery;
- Gaussian covariance transport;
- output shape and numerical validity for all released operators.

---

## Backend and Dependencies

StateVC builds on pretrained components from:

- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) for frozen speech representations.
- [kNN-VC](https://github.com/bshall/knn-vc) for the WavLM-Large loading path and prematched WavLM-conditioned HiFi-GAN synthesis backend.

These components remain frozen during StateVC conversion.

---

## Acknowledgements

We thank the authors of **WavLM** and **kNN-VC** for releasing the pretrained models and code that make this implementation possible.

If you use StateVC, please also consider citing the original WavLM and kNN-VC work.

---

## Citation

A citation entry for StateVC will be added when the corresponding paper is publicly available.

---

## Issues

For implementation questions, reproducibility problems, or bug reports, please use the repository's
[GitHub Issues](https://github.com/eurecom-asp/StateVC/issues).

---

<div align="center">


</div>
