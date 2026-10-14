import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    "madhan1705/multimodal-fake-content-models"
)

HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = BACKEND_DIR.parent

ML_MODELS_DIR = (
    PROJECT_ROOT
    / "ml-models"
)


# ============================================================
# TEXT MODEL
# ============================================================

TEXT_MODEL_DIR = (
    ML_MODELS_DIR
    / "text"
    / "roberta_base_onnx"
)


# ============================================================
# IMAGE MODEL
# ============================================================

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
# DOWNLOAD FUNCTION
# ============================================================

def download_file(
    filename: str,
    destination: Path
):
    """
    Download one model file from Hugging Face
    and copy it to the required project location.
    """

    print(
        f"\nDownloading {filename}..."
    )

    downloaded_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=filename,
        token=HF_TOKEN
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Copy downloaded file to project model directory.
    with open(
        downloaded_path,
        "rb"
    ) as source_file:

        with open(
            destination,
            "wb"
        ) as destination_file:

            while True:

                chunk = source_file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                destination_file.write(
                    chunk
                )

    print(
        f"Downloaded successfully: {destination}"
    )


# ============================================================
# DOWNLOAD ROBERTA INT8 ONNX MODEL
# ============================================================

download_file(
    "model_int8.onnx",
    TEXT_MODEL_DIR / "model_int8.onnx"
)


# ============================================================
# DOWNLOAD TOKENIZER / CONFIG FILES
# ============================================================

download_file(
    "config.json",
    TEXT_MODEL_DIR / "config.json"
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
# VERIFY REQUIRED FILES
# ============================================================

required_files = [

    # INT8 RoBERTa
    TEXT_MODEL_DIR / "model_int8.onnx",
    TEXT_MODEL_DIR / "config.json",
    TEXT_MODEL_DIR / "tokenizer.json",
    TEXT_MODEL_DIR / "tokenizer_config.json",

    # ResNet50
    IMAGE_MODEL_DIR / "resnet50_best.pth",
]


print(
    "\n"
    + "=" * 80
)

print(
    "VERIFYING DOWNLOADED MODEL FILES"
)

print(
    "=" * 80
)


for file_path in required_files:

    if not file_path.is_file():

        raise FileNotFoundError(
            f"Required model file missing: {file_path}"
        )

    file_size = file_path.stat().st_size

    print(
        f"OK: {file_path} "
        f"({file_size:,} bytes)"
    )


print(
    "\n"
    + "=" * 80
)

print(
    "ALL MODEL FILES ARE READY"
)

print(
    "=" * 80
)