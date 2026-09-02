"""Subject protocol: the model under audit."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field


@dataclass
class SubjectInfo:
    model_id: str
    family: str
    revision: str | None
    dtype: str
    quantization: str | None
    device: str
    library_versions: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "revision": self.revision,
            "resolved_revision": resolved_revision(self.model_id, self.revision),
            "dtype": self.dtype,
            "quantization": self.quantization,
            "device": self.device,
            "library_versions": self.library_versions,
            "platform": platform.platform(),
            **self.extra,
        }


def resolved_revision(model_id: str, revision: str | None = None) -> str | None:
    """The commit hash of the model files actually loaded, read from the local Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return None
    try:
        path = try_to_load_from_cache(model_id, "config.json", revision=revision)
    except Exception:  # noqa: BLE001 - offline, local path, or gated
        return None
    if not isinstance(path, str):
        return None
    parts = path.replace("\\", "/").split("/snapshots/")
    if len(parts) < 2:
        return None
    return parts[1].split("/")[0] or None


def library_versions() -> dict:
    out = {}
    for name in ("torch", "transformers", "sentence_transformers", "datasets", "numpy"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001 - optional dependency
            out[name] = None
    return out


def pick_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def gpu_snapshot() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        idx = torch.cuda.current_device()
        return {
            "gpu": torch.cuda.get_device_name(idx),
            "vram_total_gb": round(torch.cuda.get_device_properties(idx).total_memory / 2**30, 2),
            "vram_peak_gb": round(torch.cuda.max_memory_allocated(idx) / 2**30, 2),
        }
    except Exception:  # noqa: BLE001
        return {}
