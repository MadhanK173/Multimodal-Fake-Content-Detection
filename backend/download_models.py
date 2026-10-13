import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = os.getenv(
    "HF_MODEL_REPO",
    "madhan1705/multimodal-fake-content-models"
)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable is not configured."
    )


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

ML_MODELS_DIR = PROJECT_ROOT / "ml-models"

TEXT_MODEL_DIR = (
    ML_MODELS_DIR
    / "text"
    / "roberta_base_final_RECOVERED"
)

IMAGE_MODEL_DIR = (
    ML_MODELS_DIR
    / "image"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

TEXT_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE DOWNLOAD HELPER
# ============================================================

def download_file(
    filename: str,
    destination: Path
) -> None:

    print(
        f"Downloading {filename}..."
    )

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model",
        token=HF_TOKEN,
    )

    shutil.copy2(
        downloaded_path,
        destination
    )

    print(
        f"Saved to: {destination}"
    )


# ============================================================
# DOWNLOAD ROBERTA
# ============================================================

download_file(
    "config.json",
    TEXT_MODEL_DIR / "config.json"
)

download_file(
    "model.safetensors",
    TEXT_MODEL_DIR / "model.safetensors"
)

download_file(
    "tokenizer.json",
    TEXT_MODEL_DIR / "tokenizer.json"
)

download_file(
    "tokenizer_config.json",
    TEXT_MODEL_DIR / "tokenizer_config.json"
)


# ============================================================
# DOWNLOAD RESNET50
# ============================================================

download_file(
    "resnet50_best.pth",
    IMAGE_MODEL_DIR / "resnet50_best.pth"
)


# ============================================================
# VERIFY
# ============================================================

required_files = [
    TEXT_MODEL_DIR / "config.json",
    TEXT_MODEL_DIR / "model.safetensors",
    TEXT_MODEL_DIR / "tokenizer.json",
    TEXT_MODEL_DIR / "tokenizer_config.json",
    IMAGE_MODEL_DIR / "resnet50_best.pth",
]


for file_path in required_files:

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required model file missing: {file_path}"
        )


print()
print("=" * 60)
print("ALL MODEL FILES ARE READY")
print("=" * 60)