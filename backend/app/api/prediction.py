import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile
)

from ..services.multimodal_service import (
    predict_audio,
    predict_file,
    predict_image,
    predict_text
)


router = APIRouter(
    prefix="/api/v1/predict",
    tags=["Prediction"]
)


# ============================================================
# SUPPORTED FILE FORMATS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma"
}


# ============================================================
# RESPONSE STANDARDIZATION HELPERS
# ============================================================

def normalize_text_label(
    label: Any
) -> str:

    value = str(
        label or ""
    ).strip().lower()

    if value in {
        "label_0",
        "real",
        "real content"
    }:
        return "REAL CONTENT"

    if value in {
        "label_1",
        "fake",
        "fake content",
        "ai"
    }:
        return "FAKE CONTENT"

    return str(label)


def normalize_image_label(
    label: Any
) -> str:

    value = str(
        label or ""
    ).strip().lower()

    if value == "real":
        return "REAL IMAGE"

    if value in {
        "ai",
        "fake",
        "fake image"
    }:
        return "AI-GENERATED IMAGE"

    if value == "real image":
        return "REAL IMAGE"

    if value == "ai-generated image":
        return "AI-GENERATED IMAGE"

    return str(label)


def standardize_text_response(
    result: dict
) -> dict:

    raw = result.get(
        "result",
        result
    )

    prediction = normalize_text_label(
        raw.get("prediction")
    )

    return {
        "status": "success",
        "modality": "text",
        "prediction": prediction,
        "short_prediction": (
            "FAKE"
            if prediction == "FAKE CONTENT"
            else "REAL"
        ),
        "confidence": raw.get(
            "confidence"
        ),
        "confidence_percent": raw.get(
            "confidence_percent"
        ),
        "probabilities": raw.get(
            "probabilities"
        ),
        "model": raw.get(
            "model",
            "RoBERTa"
        ),
        "prediction_type": "content_authenticity",
        "raw_result": raw
    }


def standardize_image_response(
    result: dict
) -> dict:

    raw = result.get(
        "result",
        result
    )

    prediction = normalize_image_label(
        raw.get("prediction")
    )

    response = {
        "status": "success",
        "modality": "image",
        "prediction": prediction,
        "short_prediction": (
            "AI"
            if prediction == "AI-GENERATED IMAGE"
            else "REAL"
        ),
        "confidence": raw.get(
            "confidence"
        ),
        "confidence_percent": raw.get(
            "confidence_percent"
        ),
        "probabilities": raw.get(
            "probabilities"
        ),
        "model": raw.get(
            "model",
            "ResNet50"
        ),
        "prediction_type": raw.get(
            "prediction_type",
            "image_authenticity"
        )
    }

    # Preserve the three-case analysis information.
    if "analysis_case" in raw:
        response["analysis_case"] = raw[
            "analysis_case"
        ]

    if "fusion_method" in raw:
        response["fusion_method"] = raw[
            "fusion_method"
        ]

    if "ocr_text" in raw:
        response["ocr_text"] = raw[
            "ocr_text"
        ]

    if "ocr_details" in raw:
        response["ocr_details"] = raw[
            "ocr_details"
        ]

    if "visual_analysis" in raw:
        response["visual_analysis"] = raw[
            "visual_analysis"
        ]

    if "text_model_result" in raw:
        response["text_model_result"] = raw[
            "text_model_result"
        ]

    if "visual_model_result" in raw:
        response["visual_model_result"] = raw[
            "visual_model_result"
        ]

    response["raw_result"] = raw

    return response


def standardize_audio_response(
    result: dict
) -> dict:

    outer_result = result.get(
        "result",
        result
    )

    classification = outer_result.get(
        "classification",
        {}
    )

    prediction = normalize_text_label(
        classification.get(
            "prediction"
        )
    )

    return {
        "status": "success",
        "modality": "audio",
        "prediction": prediction,
        "short_prediction": (
            "FAKE"
            if prediction == "FAKE CONTENT"
            else "REAL"
        ),
        "confidence": classification.get(
            "confidence"
        ),
        "confidence_percent": classification.get(
            "confidence_percent"
        ),
        "probabilities": classification.get(
            "probabilities"
        ),
        "model": classification.get(
            "model",
            "RoBERTa"
        ),
        "transcription_model": "Whisper",
        "language": outer_result.get(
            "language"
        ),
        "transcript": outer_result.get(
            "transcript"
        ),
        "prediction_type": "content_authenticity",
        "raw_result": outer_result
    }


# ============================================================
# TEXT PREDICTION
# ============================================================

@router.post("/text")
async def text_prediction(
    text: str = Form(...)
):
    """
    Predict text using RoBERTa.
    """

    try:

        result = predict_text(
            text
        )

        return standardize_text_response(
            result
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except TypeError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Text inference failed: {exc}"
            )
        ) from exc


# ============================================================
# IMAGE PREDICTION
# ============================================================

@router.post("/image")
async def image_prediction(
    file: UploadFile = File(...)
):
    """
    Analyze an image.

    CASE 1:
        TEXT ONLY
        OCR -> RoBERTa

    CASE 2:
        IMAGE ONLY
        ResNet50

    CASE 3:
        TEXT + IMAGE
        OCR -> RoBERTa
        Image -> ResNet50
        Fusion
    """

    filename = (
        file.filename
        or ""
    )

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in IMAGE_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Supported formats: "
                + ", ".join(
                    sorted(
                        IMAGE_EXTENSIONS
                    )
                )
            )
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_path = temp_file.name

            shutil.copyfileobj(
                file.file,
                temp_file
            )

        result = predict_image(
            temp_path
        )

        response = standardize_image_response(
            result
        )

        response["filename"] = filename

        return response

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Image inference failed: {exc}"
            )
        ) from exc

    finally:

        if (
            temp_path
            and
            os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )


# ============================================================
# AUDIO PREDICTION
# ============================================================

@router.post("/audio")
async def audio_prediction(
    file: UploadFile = File(...)
):
    """
    Analyze audio using:

        Audio
          ↓
        Whisper
          ↓
        Transcript
          ↓
        RoBERTa
    """

    filename = (
        file.filename
        or ""
    )

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in AUDIO_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Supported formats: "
                + ", ".join(
                    sorted(
                        AUDIO_EXTENSIONS
                    )
                )
            )
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_path = temp_file.name

            shutil.copyfileobj(
                file.file,
                temp_file
            )

        result = predict_audio(
            temp_path
        )

        response = standardize_audio_response(
            result
        )

        response["filename"] = filename

        return response

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Audio inference failed: {exc}"
            )
        ) from exc

    finally:

        if (
            temp_path
            and
            os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )


# ============================================================
# UNIFIED IMAGE/AUDIO FILE PREDICTION
# ============================================================

@router.post("/file")
async def unified_file_prediction(
    file: UploadFile = File(...)
):
    """
    Automatically detect whether the uploaded file
    is an image or audio file.
    """

    filename = (
        file.filename
        or ""
    )

    extension = Path(
        filename
    ).suffix.lower()

    if (
        extension not in IMAGE_EXTENSIONS
        and
        extension not in AUDIO_EXTENSIONS
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Please upload an image or audio file."
            )
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_path = temp_file.name

            shutil.copyfileobj(
                file.file,
                temp_file
            )

        result = predict_file(
            temp_path
        )

        detected_modality = (
            result.get(
                "modality",
                ""
            )
            .lower()
        )

        if detected_modality == "image":

            response = standardize_image_response(
                result
            )

        elif detected_modality == "audio":

            response = standardize_audio_response(
                result
            )

        else:

            response = {
                "status": "success",
                "modality": detected_modality,
                "result": result
            }

        response["filename"] = filename

        return response

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unified inference failed: {exc}"
            )
        ) from exc

    finally:

        if (
            temp_path
            and
            os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )