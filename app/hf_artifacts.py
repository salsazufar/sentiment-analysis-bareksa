"""
Utilities for Hugging Face Spaces deployment.

This module ensures the required model artifacts exist on disk by downloading them
from a Hugging Face *Model* repository at container startup.

Expected local paths (matching current code in `app/model.py`):
  - ./best_lstm_model.keras
  - ./app/artifacts/tokenizer.pkl
  - ./app/artifacts/label_encoder.pkl

Configure via environment variables:
  - HF_MODEL_REPO_ID (required if artifacts are not already present)
  - HF_MODEL_REVISION (optional: git ref/commit/tag)
  - HF_TOKEN (optional: for private repos; can also use HUGGINGFACEHUB_API_TOKEN)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _get_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


def ensure_model_artifacts() -> None:
    """
    Ensure model + preprocessing artifacts exist on disk.

    If they are missing, download from a HF model repo into the expected paths.
    """

    root_dir = Path(__file__).resolve().parents[1]
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"

    required = {
        # remote filename -> local path
        "best_lstm_model.keras": root_dir / "best_lstm_model.keras",
        "tokenizer.pkl": artifacts_dir / "tokenizer.pkl",
        "label_encoder.pkl": artifacts_dir / "label_encoder.pkl",
    }

    missing = [str(p) for p in required.values() if not p.exists()]
    if not missing:
        return

    repo_id = os.getenv("HF_MODEL_REPO_ID")
    if not repo_id:
        raise RuntimeError(
            "Model artifacts are missing but HF_MODEL_REPO_ID is not set.\n\n"
            "Missing:\n- "
            + "\n- ".join(missing)
            + "\n\n"
            "Set HF_MODEL_REPO_ID to a Hugging Face *model* repository that contains:\n"
            "- best_lstm_model.keras\n"
            "- tokenizer.pkl\n"
            "- label_encoder.pkl\n"
        )

    revision = os.getenv("HF_MODEL_REVISION")
    token = _get_token()

    # Import only when needed (keeps local dev lightweight).
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import huggingface_hub. Please add/install 'huggingface_hub'."
        ) from e

    for remote_filename, dest in required.items():
        if dest.exists():
            continue

        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_filename,
            repo_type="model",
            revision=revision,
            token=token,
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_path, dest)
