"""Загрузка обученных экспертов и единый инференс по батчу окон.

Flow (Э1, supervised мультикласс) и AE (Э4, аномалия) объединяются здесь в один
проход: по окну [B,N,F] возвращаем вероятности классов Flow и anomaly-score AE.
Оба эксперта используют ОДИН препроцессор/энкодер (обучены на одном UNSW-сплите)
— это и есть точка консистентности фич перед мета-слоем.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mlsiem.features.preprocessing import FeaturePreprocessor
from mlsiem.models.autoencoder.model import AEConfig, Conv1DAutoencoder, reconstruction_error
from mlsiem.models.dataset import LabelEncoder
from mlsiem.models.flow_cnn_lstm.model import FlowCNNLSTM, FlowModelConfig


def _emb_dim(card: int) -> int:
    return min(16, max(2, round(card ** 0.5)))


@dataclass
class ExpertOutputs:
    flow_probs: np.ndarray   # [B, n_classes]
    ae_scores: np.ndarray    # [B]


class ExpertEnsemble:
    """Flow + AE с общим препроцессором/энкодером и AE-порогом."""

    def __init__(self, flow, ae, encoder, preprocessor, ae_threshold, classes, device):
        self.flow = flow
        self.ae = ae
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.ae_threshold = ae_threshold
        self.classes = classes
        self.device = device

    @classmethod
    def load(cls, flow_dir: str | Path, ae_dir: str | Path,
             device: torch.device | None = None, window: int = 16) -> "ExpertEnsemble":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        flow_dir, ae_dir = Path(flow_dir), Path(ae_dir)

        pre = FeaturePreprocessor.load(flow_dir / "preprocessor.json")
        enc = LabelEncoder.load(flow_dir / "label_encoder.json")
        names = pre.feature_names
        cat_positions = [names.index(c) for c in pre.vocabularies]
        cat_cards = [len(pre.vocabularies[c]) + 1 for c in pre.vocabularies]
        n_numeric = len(names) - len(cat_positions)

        flow = FlowCNNLSTM(FlowModelConfig(
            n_numeric=n_numeric, n_classes=len(enc),
            cat_cardinalities=cat_cards, cat_emb_dims=[_emb_dim(c) for c in cat_cards],
            cat_positions=cat_positions,
        )).to(device)
        flow.load_state_dict(torch.load(flow_dir / "flow_cnn_lstm.pt", map_location=device))
        flow.eval()

        ae = Conv1DAutoencoder(AEConfig(n_features=len(names), window=window)).to(device)
        ae.load_state_dict(torch.load(ae_dir / "autoencoder.pt", map_location=device))
        ae.eval()

        thr_file = ae_dir / "ae_threshold.json"
        thr = (json.loads(thr_file.read_text(encoding="utf-8"))["threshold"]
               if thr_file.exists() else None)  # None → откалибровать у вызывающего
        return cls(flow, ae, enc, pre, thr, enc.classes, device)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, mask: torch.Tensor) -> ExpertOutputs:
        x, mask = x.to(self.device), mask.to(self.device)
        flow_probs = self.flow(x, mask).softmax(dim=1).cpu().numpy()
        ae_scores = reconstruction_error(self.ae(x), x, mask).cpu().numpy()
        return ExpertOutputs(flow_probs=flow_probs, ae_scores=ae_scores)
