<div align="center">

# StateVC

### Shared-State Local Translations for Training-Free Voice Conversion

A lightweight, training-free one-shot voice conversion framework based on
**pair-specific shared acoustic states** and **local representation translation**.

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Voice%20Conversion-8A2BE2)
![Training](https://img.shields.io/badge/Training-Free-00A86B)

</div>

---

## Overview

**StateVC** is a training-free one-shot voice conversion method.

Given a source utterance and a target-speaker reference utterance, StateVC first builds a
**shared pair-specific acoustic state space** from frozen WavLM representations.

It then estimates local source-to-reference transformations within the shared states and applies
them to the source representation before waveform synthesis.

The released default system uses:

- frozen **WavLM-Large layer 6** representations;
- a **pair-specific shared diagonal GMM** for acoustic-state discovery;
- **soft state routing**;
- **state-wise mean translations** in the full WavLM feature space;
- a frozen **WavLM-conditioned HiFi-GAN** backend released with kNN-VC for synthesis.

No additional voice conversion model is trained.

---

## Highlights

- **Training-free**  
  No adaptation or additional model training is required.

- **One-shot conversion**  
  Only one source utterance and one target reference utterance are needed.

- **Shared acoustic states**  
  Source and reference are organized using the same pair-specific latent state space.

- **Local representation translation**  
  Different acoustic regions can receive different source-to-reference updates.

- **Soft routing**  
  Each source frame can combine information from multiple shared states.

- **Frozen pretrained backbone**  
  StateVC uses WavLM-Large and the kNN-VC HiFi-GAN backend without updating their parameters.

---

## Method Summary

StateVC works in three main stages:

1. **Feature extraction**  
   Extract WavLM-Large layer-6 representations from the source and reference waveforms.

2. **Shared state construction**  
   Rank WavLM dimensions according to source temporal variance, keep the top 24 routing dimensions,
   and fit a shared diagonal GMM to the pooled source and reference routing features.

3. **State-wise conversion and synthesis**  
   Estimate state-wise source and reference statistics in the full WavLM feature space,
   apply state-wise mean translation with soft routing,
   and synthesize the edited feature sequence with the frozen HiFi-GAN backend.

### Important note

The top-24 dimensions are used **only for state construction and routing**.

The actual state statistics and the final Mean operator are computed in the **full 1024-dimensional WavLM space**.

---

## Default Configuration

| Component | Setting |
|---|---|
| Representation | WavLM-Large layer 6 |
| Feature dimension | 1024 |
| Routing dimensions | 24 |
| Routing-coordinate selection | Source temporal variance |
| State model | Shared diagonal GMM |
| Candidate states | 1 to 4 |
| GMM selection | BIC |
| Minimum pooled support | 20 |
| Posterior smoothing | 5 frames |
| Minimum per-utterance state mass | 20 |
| Main operator | Mean |
| Routing | Soft |
| Conversion strength | 1.0 |
| Decoder | Frozen prematched kNN-VC HiFi-GAN |

---

## Installation

Clone the repository:

```bash
git clone git@github.com:eurecom-asp/StateVC.git
cd StateVC
