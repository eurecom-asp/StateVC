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
![Conference](https://img.shields.io/badge/ICASSP-2027-blue)

</div>

---

## Overview

**StateVC** performs one-shot voice conversion without training an additional conversion model.

Given a source utterance and a target-speaker reference utterance, StateVC first constructs a
**shared pair-specific acoustic state space** using frozen WavLM representations.

It then estimates local source-to-reference transformations within the shared states and applies
them to the source representation before waveform synthesis.

The final system uses:

- frozen **WavLM-Large layer 6** representations;
- a **pair-specific shared diagonal GMM** for acoustic-state discovery;
- **soft state routing**;
- **state-wise mean translations** in the full WavLM feature space;
- a frozen **WavLM-conditioned HiFi-GAN** released with kNN-VC for synthesis.

No VC model is trained.

---

## Highlights

- **Training-free**  
  No adaptation or additional model training is required.

- **One-shot conversion**  
  Only one source utterance and one target reference utterance are needed.

- **Shared acoustic states**  
  Source and reference frames are organized using the same pair-specific latent state space.

- **Local rather than global conversion**  
  Different acoustic regions receive different source-to-reference transformations.

- **Soft routing**  
  Each source frame can combine transformations from multiple states.

- **Frozen pretrained backbone**  
  StateVC uses WavLM-Large and the kNN-VC HiFi-GAN backend without updating their parameters.

---

## Method

```mermaid
flowchart LR
    A[Source waveform] --> C[WavLM-Large<br/>Layer 6]
    B[Reference waveform] --> D[WavLM-Large<br/>Layer 6]

    C --> E[Source-based<br/>variance ranking]
    D --> F[Shared selected<br/>routing coordinates]
    E --> F

    C --> G[Full 1024-D<br/>source features]
    D --> H[Full 1024-D<br/>reference features]

    F --> I[Shared diagonal GMM<br/>K = 1...4]
    I --> J[Posterior smoothing]
    J --> K[Soft state routing]

    G --> L[State statistics]
    H --> L

    K --> M[State-wise<br/>local translations]
    L --> M

    M --> N[Edited WavLM features]
    N --> O[Frozen HiFi-GAN]
    O --> P[Converted waveform]
