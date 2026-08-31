"""Dense reranking used to blend real and nearby-evidence results."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from lazy_grounding.schemas import SearchCandidate

_EXPECTED_EMBEDDING_DIMENSIONS = 2


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class HuggingFaceEncoder:
    """Mean-pool and L2-normalize final hidden states, matching the paper."""

    def __init__(  # pragma: no cover - exercised only with the optional model dependency
        self,
        model_name: str,
        *,
        revision: str,
        device: str = "auto",
        max_length: int = 512,
    ):
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
        from transformers import (  # type: ignore[import-not-found]  # noqa: PLC0415
            AutoModel,
            AutoTokenizer,
        )

        self._torch = torch
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = (
            AutoModel.from_pretrained(model_name, revision=revision).to(self._device).eval()
        )

    def encode(  # pragma: no cover - exercised only with the optional model dependency
        self, texts: Sequence[str]
    ) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            hidden = self._model(**encoded).last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            pooled = self._torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(pooled.cpu().numpy(), dtype=np.float32)


class DenseRanker:
    def __init__(self, encoder: Encoder):
        self._encoder = encoder
        self._cache: dict[str, NDArray[np.float32]] = {}
        self._lock = threading.Lock()

    def _encode_cached(self, texts: Sequence[str]) -> NDArray[np.float32]:
        with self._lock:
            missing = tuple(dict.fromkeys(text for text in texts if text not in self._cache))
            if missing:
                encoded = self._encoder.encode(missing)
                if encoded.ndim != _EXPECTED_EMBEDDING_DIMENSIONS or encoded.shape[0] != len(
                    missing
                ):
                    raise ValueError("Encoder returned an invalid shape")
                for text, vector in zip(missing, encoded, strict=True):
                    norm = float(np.linalg.norm(vector))
                    self._cache[text] = vector / norm if norm else np.zeros_like(vector)
            return np.stack([self._cache[text] for text in texts])

    def rank(
        self,
        query: str,
        candidates: Sequence[SearchCandidate],
        *,
        top_k: int,
    ) -> list[SearchCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []
        texts = (query, *(candidate.embedding_text() for candidate in candidates))
        vectors = self._encode_cached(texts)
        scores = vectors[1:] @ vectors[0]
        scored = [
            replace(candidate, score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(
            scored,
            key=lambda candidate: (
                -(candidate.score if candidate.score is not None else float("-inf")),
                candidate.rank_hint,
                candidate.doc_id,
            ),
        )[:top_k]
