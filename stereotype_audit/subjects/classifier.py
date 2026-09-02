"""Text classifier subject."""

from __future__ import annotations

import numpy as np
import torch

from stereotype_audit.subjects.base import SubjectInfo, gpu_snapshot, library_versions, pick_device

TARGET_LABEL_HINTS = ("toxic", "toxicity", "hate", "hateful", "offensive", "abusive", "label_1")


class ClassifierSubject:
    family = "clf"

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
        label: str | None = None,
        activation: str = "auto",
        trust_remote_code: bool = False,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        ).to(self.device)
        self.model.eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.label_index, self.label_name = self._pick_label(label)
        self.activation = self._pick_activation(activation)

    def _pick_label(self, label: str | None) -> tuple[int, str]:
        names = {v.lower(): k for k, v in self.id2label.items()}
        if label is not None:
            if label.lower() not in names:
                raise ValueError(f"label {label!r} not in {sorted(names)}")
            return names[label.lower()], label
        for hint in TARGET_LABEL_HINTS:
            for name, idx in names.items():
                if name == hint:
                    return idx, self.id2label[idx]
        for hint in TARGET_LABEL_HINTS:
            for name, idx in names.items():
                if hint in name:
                    return idx, self.id2label[idx]
        raise ValueError(f"could not guess a target label from {sorted(names)}; pass label= explicitly")

    def _pick_activation(self, activation: str) -> str:
        if activation != "auto":
            return activation
        problem = getattr(self.model.config, "problem_type", None)
        labels = {v.lower() for v in self.id2label.values()}
        if problem == "multi_label_classification" or {"obscene", "insult"} & labels or len(labels) > 6:
            return "sigmoid"
        return "softmax"

    def info(self) -> SubjectInfo:
        return SubjectInfo(
            model_id=self.model_id,
            family=self.family,
            revision=self.revision,
            dtype=str(next(self.model.parameters()).dtype).replace("torch.", ""),
            quantization=None,
            device=self.device,
            library_versions=library_versions(),
            extra={
                "target_label": self.label_name,
                "activation": self.activation,
                "labels": [self.id2label[i] for i in sorted(self.id2label)],
                **gpu_snapshot(),
            },
        )

    @torch.inference_mode()
    def scores(self, texts: list[str]) -> np.ndarray:
        """Probability of the target label for each text."""
        out = np.empty(len(texts), dtype=np.float64)
        for b in range(0, len(texts), self.batch_size):
            chunk = list(texts[b : b + self.batch_size])
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(self.device)
            logits = self.model(**enc).logits.float()
            if self.activation == "sigmoid":
                probs = torch.sigmoid(logits)[:, self.label_index]
            else:
                probs = torch.softmax(logits, dim=-1)[:, self.label_index]
            out[b : b + len(chunk)] = probs.cpu().numpy()
        return out
