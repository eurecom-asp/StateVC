from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torchaudio


KNN_VC_HUB_REF = "bshall/knn-vc:c616845c4e309e24d5927f15adbdf277a3d65358"


class KNNVCBackend:
    """Frozen WavLM-Large layer-6 + prematched HiFi-GAN backend from kNN-VC."""

    def __init__(self, device: str = "cuda", prematched: bool = True):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = torch.hub.load(
            KNN_VC_HUB_REF,
            "knn_vc",
            pretrained=True,
            prematched=prematched,
            device=device,
            trust_repo=True,
        )
        self.model.eval()
        self.sample_rate = int(self.model.sr)

    @torch.inference_mode()
    def extract(self, wav: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        """Return WavLM-Large layer-6 features [T, 1024]."""
        return self.model.get_features(wav).to(self.device)

    @torch.inference_mode()
    def vocode(self, features: torch.Tensor) -> torch.Tensor:
        """Directly synthesize a WavLM feature sequence with frozen HiFi-GAN."""
        return self.model.vocode(features[None].to(self.device)).squeeze(0).detach().cpu()

    def save(self, path: Union[str, Path], waveform: torch.Tensor) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wav = waveform.detach().cpu()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(str(path), wav, self.sample_rate)
