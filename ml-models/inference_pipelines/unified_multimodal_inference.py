import os
import sys
from pathlib import Path


class UnifiedMultimodalInferenceEngine:
    """
    UNIFIED MULTIMODAL INFERENCE ENGINE

    Supported modalities:
        - text
        - image
        - audio

    Architecture:

        TEXT
            ↓
        RoBERTa

        IMAGE
            ↓
        OCR + Image Case Detection
            ├── CASE 1: TEXT ONLY
            │       OCR → RoBERTa
            │
            ├── CASE 2: IMAGE ONLY
            │       ResNet50
            │
            └── CASE 3: TEXT + IMAGE
                    OCR → RoBERTa
                    +
                    ResNet50
                    ↓
                    Fusion

        AUDIO
            ↓
        Whisper
            ↓
        Transcript
            ↓
        RoBERTa

    IMPORTANT:
        The actual image OCR, case detection, ResNet50
        inference, probability calculation and fusion are
        implemented by ImageInferenceEngine.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        text_model_path=None,
        image_model_path=None,
        device=None,
        whisper_model_name="base"
    ):

        # ========================================================
        # DETERMINE LOCAL PROJECT ROOT
        # ========================================================

        current_dir = Path(
            __file__
        ).resolve()

        self.project_root = current_dir.parents[2]

        # ========================================================
        # LOCAL INFERENCE PIPELINE DIRECTORY
        # ========================================================

        self.pipeline_dir = (
            self.project_root
            / "ml-models"
            / "inference_pipelines"
        )

        if not self.pipeline_dir.is_dir():
            raise FileNotFoundError(
                "Inference pipeline directory not found:\n"
                f"{self.pipeline_dir}"
            )

        pipeline_dir_string = str(
            self.pipeline_dir
        )

        if pipeline_dir_string not in sys.path:
            sys.path.insert(
                0,
                pipeline_dir_string
            )

        # ========================================================
        # DEFAULT TEXT MODEL
        # ========================================================

        if text_model_path is None:
            text_model_path = (
                self.project_root
                / "ml-models"
                / "text"
                / "roberta_base_final_RECOVERED"
            )

        # ========================================================
        # DEFAULT IMAGE MODEL
        # ========================================================

        if image_model_path is None:
            image_model_path = (
                self.project_root
                / "ml-models"
                / "image"
                / "resnet50_best.pth"
            )

        self.text_model_path = Path(
            text_model_path
        ).resolve()

        self.image_model_path = Path(
            image_model_path
        ).resolve()

        # ========================================================
        # VALIDATE MODEL PATHS
        # ========================================================

        if not self.text_model_path.is_dir():
            raise FileNotFoundError(
                "RoBERTa model directory not found:\n"
                f"{self.text_model_path}"
            )

        if not self.image_model_path.is_file():
            raise FileNotFoundError(
                "ResNet50 model file not found:\n"
                f"{self.image_model_path}"
            )

        self.device = device

        # ========================================================
        # IMPORT LOCAL INFERENCE ENGINES
        # ========================================================

        try:

            from text_inference import (
                TextInferenceEngine
            )

            from image_inference import (
                ImageInferenceEngine
            )

            from audio_inference import (
                AudioInferenceEngine
            )

        except Exception as exc:

            raise RuntimeError(
                "Failed to import local inference pipelines.\n"
                f"Pipeline directory: {self.pipeline_dir}\n"
                f"Original error: {exc}"
            ) from exc

        # ========================================================
        # INITIALIZE TEXT ENGINE
        # ========================================================

        print(
            "=" * 70
        )

        print(
            "INITIALIZING TEXT INFERENCE ENGINE"
        )

        print(
            "=" * 70
        )

        self.text_engine = TextInferenceEngine(
            model_path=str(
                self.text_model_path
            ),
            max_length=512,
            device=self.device
        )

        print(
            "TextInferenceEngine initialized."
        )

        # ========================================================
        # INITIALIZE IMAGE ENGINE
        # ========================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "INITIALIZING IMAGE INFERENCE ENGINE"
        )

        print(
            "=" * 70
        )

        self.image_engine = ImageInferenceEngine(
            model_path=str(
                self.image_model_path
            ),
            text_model_path=str(
                self.text_model_path
            ),
            device=self.device,
            image_size=224
        )

        print(
            "ImageInferenceEngine initialized."
        )

        print(
            "Visual model: ResNet50"
        )

        # ========================================================
        # INITIALIZE AUDIO ENGINE
        # ========================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "INITIALIZING AUDIO INFERENCE ENGINE"
        )

        print(
            "=" * 70
        )

        self.audio_engine = AudioInferenceEngine(
            whisper_model_name=whisper_model_name,
            device=self.device
        )

        print(
            "AudioInferenceEngine initialized."
        )

        # ========================================================
        # FINAL STATUS
        # ========================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "UNIFIED MULTIMODAL INFERENCE ENGINE READY"
        )

        print(
            "=" * 70
        )

    # ================================================================
    # TEXT PREDICTION
    # ================================================================

    def predict_text(
        self,
        text
    ):

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

        result = self.text_engine.predict(
            text
        )

        return {
            "modality": "text",
            "input_type": "text",
            "trained_model_name": "RoBERTa",
            "model_name": "RoBERTa",
            "result": result
        }

    # ================================================================
    # IMAGE RESULT NORMALIZATION
    # ================================================================

    def _normalize_image_result(
        self,
        result
    ):
        """
        Preserve the complete result returned by
        ImageInferenceEngine while adding stable image-model
        metadata.

        IMPORTANT:
            This method does NOT invent OCR text,
            probabilities, confidence values, image cases,
            or fusion results.

        Those values must come from ImageInferenceEngine.
        """

        if not isinstance(
            result,
            dict
        ):
            raise TypeError(
                "ImageInferenceEngine must return a dictionary."
            )

        normalized_result = dict(
            result
        )

        # --------------------------------------------------------
        # IMAGE MODEL
        # --------------------------------------------------------

        normalized_result[
            "trained_model_name"
        ] = "ResNet50"

        normalized_result[
            "model_name"
        ] = "ResNet50"

        normalized_result[
            "visual_model_name"
        ] = "ResNet50"

        # --------------------------------------------------------
        # MODALITY
        # --------------------------------------------------------

        normalized_result[
            "modality"
        ] = "image"

        # --------------------------------------------------------
        # VERIFICATION
        # --------------------------------------------------------

        normalized_result[
            "verification_enabled"
        ] = False

        normalized_result.setdefault(
            "web_verification",
            {
                "enabled": False,
                "status": "NOT_APPLICABLE",
                "message": (
                    "Image authenticity is determined "
                    "by the trained image model."
                )
            }
        )

        return normalized_result

    # ================================================================
    # IMAGE PREDICTION
    # ================================================================

    def predict_image(
        self,
        image_path
    ):
        """
        Run image through the ImageInferenceEngine.

        ImageInferenceEngine is responsible for:

            IMAGE
              ↓
            OCR
              ↓
            CASE DETECTION
              ↓
            ┌─────────────────────────────┐
            │ CASE 1 → TEXT ONLY          │
            │ CASE 2 → IMAGE ONLY         │
            │ CASE 3 → TEXT + IMAGE       │
            └─────────────────────────────┘
              ↓
            ResNet50
              ↓
            Fusion when required
              ↓
            Final prediction
        """

        if not isinstance(
            image_path,
            (str, os.PathLike)
        ):
            raise TypeError(
                "Image input must be a file path."
            )

        image_path = Path(
            image_path
        ).resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found:\n{image_path}"
            )

        # --------------------------------------------------------
        # RUN ACTUAL IMAGE PIPELINE
        # --------------------------------------------------------

        result = self.image_engine.predict(
            str(image_path)
        )

        # --------------------------------------------------------
        # NORMALIZE ONLY METADATA
        # --------------------------------------------------------

        normalized_result = (
            self._normalize_image_result(
                result
            )
        )

        return {
            "modality": "image",
            "input_type": "image_file",
            "input_path": str(
                image_path
            ),

            # Explicit image model metadata
            "trained_model_name": "ResNet50",
            "model_name": "ResNet50",
            "visual_model_name": "ResNet50",

            # Keep complete original image result
            "result": normalized_result
        }

    # ================================================================
    # AUDIO PREDICTION
    # ================================================================

    def predict_audio(
        self,
        audio_path
    ):

        if not isinstance(
            audio_path,
            (str, os.PathLike)
        ):
            raise TypeError(
                "Audio input must be a file path."
            )

        audio_path = Path(
            audio_path
        ).resolve()

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found:\n{audio_path}"
            )

        result = self.audio_engine.predict(
            str(audio_path)
        )

        return {
            "modality": "audio",
            "input_type": "audio_file",
            "input_path": str(
                audio_path
            ),
            "trained_model_name": "Whisper + RoBERTa",
            "model_name": "Whisper + RoBERTa",
            "result": result
        }

    # ================================================================
    # AUTOMATIC MODALITY DETECTION
    # ================================================================

    def detect_modality(
        self,
        input_data
    ):

        if not isinstance(
            input_data,
            (str, os.PathLike)
        ):
            raise TypeError(
                "Unsupported input type."
            )

        # --------------------------------------------------------
        # PATH-LIKE INPUT
        # --------------------------------------------------------

        if isinstance(
            input_data,
            os.PathLike
        ):

            input_path = Path(
                input_data
            ).resolve()

            if not input_path.is_file():
                raise FileNotFoundError(
                    f"Input file not found:\n{input_path}"
                )

            return self._detect_file_modality(
                input_path
            )

        # --------------------------------------------------------
        # STRING INPUT
        # --------------------------------------------------------

        input_string = input_data

        possible_path = Path(
            input_string
        )

        if possible_path.is_file():

            return self._detect_file_modality(
                possible_path.resolve()
            )

        # Otherwise it is text.
        return "text"

    # ================================================================
    # FILE MODALITY DETECTION
    # ================================================================

    def _detect_file_modality(
        self,
        file_path
    ):

        extension = file_path.suffix.lower()

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
            ".tif",
            ".tiff"
        }

        audio_extensions = {
            ".wav",
            ".mp3",
            ".m4a",
            ".flac",
            ".ogg",
            ".aac",
            ".wma",
            ".webm"
        }

        if extension in image_extensions:
            return "image"

        if extension in audio_extensions:
            return "audio"

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    # ================================================================
    # UNIFIED PREDICTION
    # ================================================================

    def predict(
        self,
        input_data,
        modality="auto"
    ):

        if not isinstance(
            modality,
            str
        ):
            raise TypeError(
                "Modality must be a string."
            )

        modality = modality.strip().lower()

        if modality == "auto":

            modality = self.detect_modality(
                input_data
            )

        # --------------------------------------------------------
        # TEXT
        # --------------------------------------------------------

        if modality == "text":

            return self.predict_text(
                input_data
            )

        # --------------------------------------------------------
        # IMAGE
        # --------------------------------------------------------

        if modality == "image":

            return self.predict_image(
                input_data
            )

        # --------------------------------------------------------
        # AUDIO
        # --------------------------------------------------------

        if modality == "audio":

            return self.predict_audio(
                input_data
            )

        # --------------------------------------------------------
        # INVALID MODALITY
        # --------------------------------------------------------

        raise ValueError(
            "Unsupported modality. "
            "Use: text, image, audio, or auto."
        )