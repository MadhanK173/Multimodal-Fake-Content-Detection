import importlib.util
import os
from pathlib import Path
from threading import Lock
from typing import Any


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

# Expected structure:
#
# services/
#   app/
#   backend/
#   Multimodal_Fake_Content_Detection/
#
#   ml-models/
#       inference_pipelines/
#           unified_multimodal_inference.py

PROJECT_ROOT = CURRENT_FILE.parents[3]

PIPELINE_DIR = (
    PROJECT_ROOT
    / "ml-models"
    / "inference_pipelines"
)

UNIFIED_ENGINE_FILE = (
    PIPELINE_DIR
    / "unified_multimodal_inference.py"
)


# ============================================================
# VALIDATE PIPELINE PATH
# ============================================================

if not PIPELINE_DIR.is_dir():
    raise FileNotFoundError(
        "Inference pipeline directory not found:\n"
        f"{PIPELINE_DIR}"
    )


if not UNIFIED_ENGINE_FILE.is_file():
    raise FileNotFoundError(
        "Unified inference engine file not found:\n"
        f"{UNIFIED_ENGINE_FILE}"
    )


# ============================================================
# DYNAMICALLY LOAD UNIFIED ENGINE
# ============================================================

def _load_unified_engine_class():
    """
    Load UnifiedMultimodalInferenceEngine directly from the
    local Python file.

    This avoids import-resolution problems caused by the
    ml-models directory name and dynamically configured
    inference-pipeline path.
    """

    pipeline_path = str(PIPELINE_DIR)

    if pipeline_path not in os.sys.path:
        os.sys.path.insert(
            0,
            pipeline_path
        )

    module_name = (
        "local_unified_multimodal_inference"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(UNIFIED_ENGINE_FILE)
    )

    if spec is None:
        raise ImportError(
            "Could not create module specification for:\n"
            f"{UNIFIED_ENGINE_FILE}"
        )

    if spec.loader is None:
        raise ImportError(
            "Could not create module loader for:\n"
            f"{UNIFIED_ENGINE_FILE}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    engine_class = getattr(
        module,
        "UnifiedMultimodalInferenceEngine",
        None
    )

    if engine_class is None:
        raise ImportError(
            "UnifiedMultimodalInferenceEngine was not found in:\n"
            f"{UNIFIED_ENGINE_FILE}"
        )

    return engine_class


# ============================================================
# LOAD ENGINE CLASS
# ============================================================

UnifiedMultimodalInferenceEngine = (
    _load_unified_engine_class()
)


# ============================================================
# SINGLE ENGINE INSTANCE
# ============================================================

_engine = None
_engine_lock = Lock()


def get_engine():
    """
    Create the unified ML engine only once.

    This prevents every API request from loading
    RoBERTa, ResNet50 and Whisper again.
    """

    global _engine

    if _engine is None:

        with _engine_lock:

            if _engine is None:

                print("=" * 80)
                print("INITIALIZING MULTIMODAL ML ENGINE")
                print("=" * 80)

                print(
                    f"Project root:\n{PROJECT_ROOT}"
                )

                print(
                    f"Pipeline directory:\n{PIPELINE_DIR}"
                )

                print(
                    f"Unified inference engine:\n"
                    f"{UNIFIED_ENGINE_FILE}"
                )

                _engine = (
                    UnifiedMultimodalInferenceEngine()
                )

                print("=" * 80)
                print("MULTIMODAL ML ENGINE READY")
                print("=" * 80)

    return _engine


# ============================================================
# TEXT PREDICTION
# ============================================================

def predict_text(
    text: str
) -> dict[str, Any]:
    """
    Run text prediction through the unified engine.
    """

    if not isinstance(
        text,
        str
    ):
        raise TypeError(
            "Text input must be a string."
        )

    if not text.strip():
        raise ValueError(
            "Text input cannot be empty."
        )

    engine = get_engine()

    result = engine.predict_text(
        text
    )

    if not isinstance(
        result,
        dict
    ):
        raise TypeError(
            "Text prediction engine must return a dictionary."
        )

    return result


# ============================================================
# IMAGE RESULT NORMALIZATION
# ============================================================

def _normalize_image_result(
    result: dict[str, Any]
) -> dict[str, Any]:
    """
    Normalize the image result returned by the unified engine.

    IMPORTANT:
    This function does NOT create fake model probabilities,
    fake OCR text, or fake case classifications.

    It only adds stable metadata that identifies the visual
    model used by the image pipeline.

    The actual OCR, case detection, ResNet50 prediction,
    multimodal fusion and probabilities must come from the
    unified inference engine itself.
    """

    normalized = dict(result)

    # --------------------------------------------------------
    # IMAGE MODEL NAME
    # --------------------------------------------------------
    #
    # The visual image model in this project is ResNet50.
    #
    # We keep the original result intact and expose stable
    # names for the backend/frontend.
    #

    normalized[
        "trained_model_name"
    ] = "ResNet50"

    normalized[
        "model_name"
    ] = "ResNet50"

    normalized[
        "visual_model_name"
    ] = "ResNet50"

    # --------------------------------------------------------
    # MODALITY
    # --------------------------------------------------------

    normalized[
        "modality"
    ] = "image"

    # --------------------------------------------------------
    # VERIFICATION
    # --------------------------------------------------------
    #
    # Image authenticity is determined by the trained image
    # pipeline. Live web verification must not override it.
    #

    normalized[
        "verification_enabled"
    ] = False

    # Only add this if the engine did not already provide it.
    normalized.setdefault(
        "web_verification",
        {
            "status": "NOT_APPLICABLE",
            "enabled": False,
            "message": (
                "Image authenticity is determined "
                "by the trained image model."
            ),
        }
    )

    # --------------------------------------------------------
    # PRESERVE EXISTING IMAGE PIPELINE INFORMATION
    # --------------------------------------------------------
    #
    # Do not overwrite these fields if the unified engine
    # already provides them:
    #
    # - image_type
    # - analysis_case
    # - case
    # - ocr_text
    # - extracted_text
    # - text_prediction
    # - visual_prediction
    # - fusion
    # - probabilities
    # - confidence
    #
    # They must come from the actual inference pipeline.
    #

    return normalized


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(
    image_path: str
) -> dict[str, Any]:
    """
    Run image prediction through the unified multimodal engine.

    The unified engine is responsible for:

        Image
          ↓
        OCR
          ↓
        Case detection
          ↓
        Case 1 / Case 2 / Case 3
          ↓
        ResNet50
          ↓
        Multimodal fusion when required
          ↓
        Final image prediction

    This wrapper does not manufacture OCR, probabilities,
    confidence values, or analysis cases.
    """

    if not isinstance(
        image_path,
        str
    ):
        raise TypeError(
            "Image path must be a string."
        )

    if not image_path.strip():
        raise ValueError(
            "Image path cannot be empty."
        )

    if not os.path.isfile(
        image_path
    ):
        raise FileNotFoundError(
            "Image file not found:\n"
            f"{image_path}"
        )

    engine = get_engine()

    result = engine.predict_image(
        image_path
    )

    if not isinstance(
        result,
        dict
    ):
        raise TypeError(
            "Image prediction engine must return a dictionary."
        )

    return _normalize_image_result(
        result
    )


# ============================================================
# AUDIO PREDICTION
# ============================================================

def predict_audio(
    audio_path: str
) -> dict[str, Any]:
    """
    Run audio prediction through the unified engine.
    """

    if not isinstance(
        audio_path,
        str
    ):
        raise TypeError(
            "Audio path must be a string."
        )

    if not audio_path.strip():
        raise ValueError(
            "Audio path cannot be empty."
        )

    if not os.path.isfile(
        audio_path
    ):
        raise FileNotFoundError(
            "Audio file not found:\n"
            f"{audio_path}"
        )

    engine = get_engine()

    result = engine.predict_audio(
        audio_path
    )

    if not isinstance(
        result,
        dict
    ):
        raise TypeError(
            "Audio prediction engine must return a dictionary."
        )

    return result


# ============================================================
# UNIFIED FILE PREDICTION
# ============================================================

def predict_file(
    file_path: str
) -> dict[str, Any]:
    """
    Automatically detect the modality and run the
    corresponding prediction pipeline.
    """

    if not isinstance(
        file_path,
        str
    ):
        raise TypeError(
            "File path must be a string."
        )

    if not file_path.strip():
        raise ValueError(
            "File path cannot be empty."
        )

    if not os.path.isfile(
        file_path
    ):
        raise FileNotFoundError(
            "Input file not found:\n"
            f"{file_path}"
        )

    engine = get_engine()

    result = engine.predict(
        file_path,
        modality="auto"
    )

    if not isinstance(
        result,
        dict
    ):
        raise TypeError(
            "Unified prediction engine must return a dictionary."
        )

    # If the unified engine identifies this as an image,
    # apply the same image metadata normalization.
    modality = str(
        result.get(
            "modality",
            result.get(
                "type",
                ""
            )
        )
    ).strip().lower()

    if modality == "image":
        return _normalize_image_result(
            result
        )

    return result