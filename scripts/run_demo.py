"""Audit the demo subjects and copy their cards into docs/cards.

Waits for enough free VRAM before loading each full-size language model so the
run can be started while the GPU is busy with something else.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# fewer allocator fragmentation failures on a 16 GB card; must be set before torch is imported
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stereotype_audit.run import audit  # noqa: E402

SUBJECTS = [
    # (model id, family, batteries, extra battery config, needs_gb)
    (
        "ibm-granite/granite-4.2-3b",
        "lm",
        ["assoc", "decision", "nameswap", "chatiat", "bbq", "crows", "contamination", "effort"],
        {},
        9.0,
    ),
    (
        "Qwen/Qwen3.5-4B",
        "lm",
        ["assoc", "decision", "nameswap", "chatiat", "bbq", "crows", "contamination"],
        {},
        11.0,
    ),
    ("ibm-granite/granite-embedding-311m-multilingual-r2", "embed", ["seat", "retrieval"], {}, 2.0),
    ("Qwen/Qwen3-Embedding-0.6B", "embed", ["seat", "retrieval"], {}, 3.0),
    ("unitary/toxic-bert", "clf", ["ctf"], {}, 2.0),
    ("unitary/unbiased-toxic-roberta", "clf", ["ctf"], {}, 2.0),
]


def free_vram_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


def wait_for_vram(needed: float, poll_s: int = 60, log=print) -> None:
    while True:
        free = free_vram_gb()
        if free >= needed:
            return
        log(f"waiting for GPU: {free:.1f} GB free, need {needed:.1f} GB")
        time.sleep(poll_s)


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated substrings of model ids to run")
    ap.add_argument("--nulls", action="store_true")
    ap.add_argument("--n", type=int)
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--cards", default=str(ROOT / "docs" / "cards"))
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    for model_id, family, batteries, cfg, needs in SUBJECTS:
        if args.only and not any(s in model_id for s in args.only.split(",")):
            continue
        if not args.no_wait:
            wait_for_vram(needs, log=log)
        out_dir = Path(args.out) / slug(model_id)
        t0 = time.time()
        log(f"start {model_id}")
        audit(
            model_id,
            out_dir,
            batteries=batteries,
            family=family,
            n=args.n,
            nulls=args.nulls,
            battery_cfg=cfg,
            log=log,
            resume=True,
        )
        log(f"done {model_id} in {time.time() - t0:.0f} s")
        dest = Path(args.cards) / slug(model_id)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("card.json", "card.md", "card.html"):
            shutil.copyfile(out_dir / name, dest / name)
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
