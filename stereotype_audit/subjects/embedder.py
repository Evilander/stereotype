"""Embedding model subject."""

from __future__ import annotations

import numpy as np

from stereotype_audit.subjects.base import SubjectInfo, gpu_snapshot, library_versions, pick_device


class EmbedderSubject:
    family = "embed"

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
        trust_remote_code: bool = False,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.revision = revision
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_id, revision=revision, device=self.device, trust_remote_code=trust_remote_code
        )
        self.prompts = dict(getattr(self.model, "prompts", {}) or {})

    def info(self) -> SubjectInfo:
        return SubjectInfo(
            model_id=self.model_id,
            family=self.family,
            revision=self.revision,
            dtype=str(next(self.model.parameters()).dtype).replace("torch.", ""),
            quantization=None,
            device=self.device,
            library_versions=library_versions(),
            extra={"prompt_names": sorted(self.prompts), **gpu_snapshot()},
        )

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        """Unit-normalised embeddings. `kind` selects a model-defined prompt (e.g. "query") when one exists."""
        kwargs = {"batch_size": self.batch_size, "normalize_embeddings": True, "convert_to_numpy": True}
        if kind in self.prompts:
            kwargs["prompt_name"] = kind
        return np.asarray(self.model.encode(list(texts), **kwargs), dtype=np.float64)
