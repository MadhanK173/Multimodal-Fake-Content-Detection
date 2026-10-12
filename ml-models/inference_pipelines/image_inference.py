import os
import re

import torch
import torch.nn as nn
import numpy as np
import cv2
import pytesseract

from PIL import Image
from pytesseract import Output
from torchvision import models, transforms


class ImageInferenceEngine:
    """
    MULTIMODAL IMAGE INFERENCE ENGINE

    CASE 1:
        TEXT ONLY
        OCR -> RoBERTa

    CASE 2:
        IMAGE ONLY
        ResNet50 -> REAL / AI

    CASE 3:
        TEXT + IMAGE
        OCR -> RoBERTa
        Image -> ResNet50
        Fusion -> Final Prediction

    IMPORTANT:
        The image model is ResNet50.

        OCR is used only to determine whether meaningful text
        is present and, when required, to pass that text to
        RoBERTa.

        Image probabilities always come from ResNet50 softmax.

        Final probabilities are always normalized so that:

            REAL + AI = 100%

        Final model confidence is always the probability of
        the predicted class.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        model_path=None,
        text_model_path=None,
        device=None,
        image_size=224
    ):

        # ========================================================
        # LOCAL PROJECT PATHS
        # ========================================================

        current_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        project_root = os.path.abspath(
            os.path.join(
                current_dir,
                "..",
                ".."
            )
        )

        # ========================================================
        # DEFAULT RESNET50 PATH
        # ========================================================

        if model_path is None:

            model_path = os.path.join(
                project_root,
                "ml-models",
                "image",
                "resnet50_best.pth"
            )

        # ========================================================
        # DEFAULT ROBERTA PATH
        # ========================================================

        if text_model_path is None:

            text_model_path = os.path.join(
                project_root,
                "ml-models",
                "text",
                "roberta_base_final_RECOVERED"
            )

        self.model_path = os.path.abspath(
            model_path
        )

        self.text_model_path = os.path.abspath(
            text_model_path
        )

        # ========================================================
        # VALIDATE MODEL PATHS
        # ========================================================

        if not os.path.isfile(
            self.model_path
        ):

            raise FileNotFoundError(
                "ResNet50 model not found:\n"
                f"{self.model_path}"
            )

        if not os.path.isdir(
            self.text_model_path
        ):

            raise FileNotFoundError(
                "RoBERTa model directory not found:\n"
                f"{self.text_model_path}"
            )

        # ========================================================
        # CONFIGURE TESSERACT
        # ========================================================

        self._configure_tesseract()

        # ========================================================
        # DEVICE
        # ========================================================

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        self.image_size = image_size

        print(
            f"Image inference device: {self.device}"
        )

        # ========================================================
        # CLASS MAPPING
        # ========================================================
        #
        # KEEP THIS MAPPING CONSISTENT WITH YOUR TRAINED
        # RESNET50 MODEL.
        #
        # Existing project mapping:
        #
        #   0 -> AI
        #   1 -> REAL
        #

        self.class_mapping = {
            0: "ai",
            1: "real"
        }

        self.display_mapping = {
            "ai": "AI-GENERATED IMAGE",
            "real": "REAL IMAGE"
        }

        # ========================================================
        # LOAD RESNET50
        # ========================================================

        print(
            "Loading ResNet50 Image Model..."
        )

        self.model = models.resnet50(
            weights=None
        )

        self.model.fc = nn.Linear(
            2048,
            2
        )

        checkpoint = self._load_checkpoint(
            self.model_path
        )

        if (
            isinstance(
                checkpoint,
                dict
            )
            and
            "model_state_dict" in checkpoint
        ):

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "ResNet50 loaded successfully."
        )

        # ========================================================
        # IMAGE PREPROCESSING
        # ========================================================

        self.transform = transforms.Compose([
            transforms.Resize(
                (
                    self.image_size,
                    self.image_size
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        # ========================================================
        # LOAD ROBERTA
        # ========================================================

        print(
            "Loading RoBERTa Text Model for OCR content..."
        )

        from text_inference import TextInferenceEngine

        self.text_engine = TextInferenceEngine(
            model_path=self.text_model_path,
            max_length=512,
            device=self.device
        )

        print(
            "RoBERTa Text Model loaded successfully."
        )

        print(
            "Multimodal ImageInferenceEngine initialized successfully."
        )

    # ============================================================
    # TESSERACT CONFIGURATION
    # ============================================================

    def _configure_tesseract(self):

        if os.name == "nt":

            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ]

            for path in possible_paths:

                if os.path.isfile(
                    path
                ):

                    pytesseract.pytesseract.tesseract_cmd = (
                        path
                    )

                    print(
                        f"Tesseract configured:\n{path}"
                    )

                    return

        try:

            version = (
                pytesseract
                .get_tesseract_version()
            )

            print(
                f"Tesseract available: {version}"
            )

        except Exception as exc:

            raise RuntimeError(
                "Tesseract OCR executable was not found. "
                "Install Tesseract OCR or add it to PATH.\n"
                f"Original error: {exc}"
            ) from exc

    # ============================================================
    # SAFE CHECKPOINT LOADING
    # ============================================================

    def _load_checkpoint(
        self,
        model_path
    ):

        try:

            return torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )

        except TypeError:

            return torch.load(
                model_path,
                map_location=self.device
            )

    # ============================================================
    # OCR IMAGE PREPROCESSING
    # ============================================================

    def _create_ocr_variants(
        self,
        image
    ):
        """
        Create several OCR-friendly image versions.

        Tesseract documentation recommends appropriate
        preprocessing/rescaling when OCR quality is poor.
        """

        variants = []

        height, width = image.shape[:2]

        # --------------------------------------------------------
        # ORIGINAL
        # --------------------------------------------------------

        variants.append(
            (
                "original",
                image
            )
        )

        # --------------------------------------------------------
        # UPSCALED GRAYSCALE
        # --------------------------------------------------------

        scale = 2.0

        upscaled = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(
            upscaled,
            cv2.COLOR_BGR2GRAY
        )

        variants.append(
            (
                "upscaled_gray",
                gray
            )
        )

        # --------------------------------------------------------
        # CLAHE ENHANCEMENT
        # --------------------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(
            gray
        )

        variants.append(
            (
                "clahe",
                enhanced
            )
        )

        # --------------------------------------------------------
        # ADAPTIVE THRESHOLD
        # --------------------------------------------------------

        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        variants.append(
            (
                "adaptive_threshold",
                adaptive
            )
        )

        # --------------------------------------------------------
        # OTSU
        # --------------------------------------------------------

        _, otsu = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )

        variants.append(
            (
                "otsu",
                otsu
            )
        )

        return variants

    # ============================================================
    # CLEAN OCR TEXT
    # ============================================================

    def _clean_ocr_text(
        self,
        text
    ):

        if not text:
            return ""

        text = str(
            text
        )

        # Remove control characters.
        text = re.sub(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
            " ",
            text
        )

        # Normalize whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        text = text.strip()

        return text

    # ============================================================
    # OCR TOKEN QUALITY
    # ============================================================

    def _calculate_text_quality(
        self,
        text,
        confidences
    ):
        """
        Estimate whether OCR output is meaningful enough to
        be used for case detection.

        This is intentionally conservative so random visual
        patterns from photographs are less likely to become
        false OCR text.
        """

        if not text:

            return 0.0

        tokens = re.findall(
            r"\S+",
            text
        )

        if not tokens:

            return 0.0

        valid_tokens = 0

        for token in tokens:

            cleaned = re.sub(
                r"[^A-Za-z0-9À-ÖØ-öø-ÿ]",
                "",
                token
            )

            if len(cleaned) >= 2:

                valid_tokens += 1

        token_quality = (
            valid_tokens
            /
            max(
                len(tokens),
                1
            )
        )

        if confidences:

            confidence_score = (
                float(
                    np.mean(
                        confidences
                    )
                )
                /
                100.0
            )

        else:

            confidence_score = 0.0

        # Length gives only a small contribution.
        length_score = min(
            len(text) / 120.0,
            1.0
        )

        quality = (
            confidence_score * 0.55
            +
            token_quality * 0.30
            +
            length_score * 0.15
        )

        return float(
            max(
                0.0,
                min(
                    quality,
                    1.0
                )
            )
        )

    # ============================================================
    # RUN OCR ON ONE IMAGE
    # ============================================================

    def _run_single_ocr(
        self,
        image,
        psm
    ):

        config = (
            f"--oem 3 --psm {psm}"
        )

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config=config
        )

        words = []
        boxes = []
        confidences = []

        image_height, image_width = image.shape[:2]

        for i in range(
            len(
                data["text"]
            )
        ):

            raw_text = str(
                data["text"][i]
            ).strip()

            try:

                confidence = float(
                    data["conf"][i]
                )

            except (
                ValueError,
                TypeError
            ):

                confidence = -1.0

            # More conservative than the old 30 threshold.
            if (
                not raw_text
                or
                confidence < 45
            ):
                continue

            words.append(
                raw_text
            )

            confidences.append(
                confidence
            )

            x = int(
                data["left"][i]
            )

            y = int(
                data["top"][i]
            )

            w = int(
                data["width"][i]
            )

            h = int(
                data["height"][i]
            )

            boxes.append(
                (
                    x,
                    y,
                    w,
                    h
                )
            )

        extracted_text = self._clean_ocr_text(
            " ".join(
                words
            )
        )

        text_area = 0

        for (
            x,
            y,
            w,
            h
        ) in boxes:

            text_area += (
                w * h
            )

        image_area = (
            image_width
            *
            image_height
        )

        if image_area > 0:

            text_area_ratio = (
                text_area
                /
                image_area
            )

        else:

            text_area_ratio = 0.0

        if confidences:

            average_confidence = float(
                np.mean(
                    confidences
                )
            )

        else:

            average_confidence = 0.0

        text_quality = (
            self._calculate_text_quality(
                extracted_text,
                confidences
            )
        )

        return {
            "text": extracted_text,
            "word_count": len(words),
            "character_count": len(extracted_text),
            "boxes": boxes,
            "text_area_ratio": float(
                text_area_ratio
            ),
            "ocr_confidence": float(
                average_confidence
            ),
            "text_quality": float(
                text_quality
            ),
            "psm": psm
        }

    # ============================================================
    # OCR TEXT EXTRACTION
    # ============================================================

    def extract_text_from_image(
        self,
        image_path
    ):

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        original_height, original_width = (
            image.shape[:2]
        )

        variants = (
            self._create_ocr_variants(
                image
            )
        )

        candidates = []

        # Use segmentation modes suited to blocks and sparse text.
        #
        # Tesseract documents PSM 6 as a uniform text block
        # and PSM 11 as sparse text. 
        for (
            variant_name,
            variant_image
        ) in variants:

            for psm in (
                6,
                11
            ):

                try:

                    result = (
                        self._run_single_ocr(
                            variant_image,
                            psm
                        )
                    )

                except Exception:

                    continue

                result[
                    "variant"
                ] = variant_name

                candidates.append(
                    result
                )

        if not candidates:

            return {
                "text": "",
                "word_count": 0,
                "character_count": 0,
                "boxes": [],
                "text_area_ratio": 0.0,
                "ocr_confidence": 0.0,
                "text_quality": 0.0,
                "ocr_reliable": False,
                "ocr_variant": None,
                "ocr_psm": None
            }

        # --------------------------------------------------------
        # SCORE OCR CANDIDATES
        # --------------------------------------------------------

        for candidate in candidates:

            confidence_score = (
                candidate[
                    "ocr_confidence"
                ]
                /
                100.0
            )

            word_score = min(
                candidate[
                    "word_count"
                ]
                /
                12.0,
                1.0
            )

            quality_score = candidate[
                "text_quality"
            ]

            coverage_score = min(
                candidate[
                    "text_area_ratio"
                ]
                /
                0.08,
                1.0
            )

            candidate[
                "_selection_score"
            ] = (
                quality_score * 0.50
                +
                confidence_score * 0.30
                +
                word_score * 0.10
                +
                coverage_score * 0.10
            )

        best = max(
            candidates,
            key=lambda item: item[
                "_selection_score"
            ]
        )

        # --------------------------------------------------------
        # OCR RELIABILITY
        # --------------------------------------------------------

        text = best[
            "text"
        ]

        word_count = best[
            "word_count"
        ]

        character_count = best[
            "character_count"
        ]

        confidence = best[
            "ocr_confidence"
        ]

        quality = best[
            "text_quality"
        ]

        text_area_ratio = best[
            "text_area_ratio"
        ]

        # Conservative reliability test.
        #
        # This prevents random letters detected inside a
        # photograph from automatically becoming Case 3.
        reliable = (
            character_count >= 15
            and
            word_count >= 3
            and
            confidence >= 50.0
            and
            quality >= 0.48
            and
            (
                text_area_ratio >= 0.008
                or
                word_count >= 8
            )
        )

        # --------------------------------------------------------
        # RESCALE BOXES BACK TO ORIGINAL IMAGE SIZE
        # --------------------------------------------------------

        variant_name = best[
            "variant"
        ]

        if variant_name == "original":

            scale_x = 1.0
            scale_y = 1.0

        else:

            scale_x = (
                original_width
                /
                max(
                    variant_image.shape[1],
                    1
                )
            )

            scale_y = (
                original_height
                /
                max(
                    variant_image.shape[0],
                    1
                )
            )

        scaled_boxes = []

        for (
            x,
            y,
            w,
            h
        ) in best["boxes"]:

            scaled_boxes.append(
                (
                    int(
                        x * scale_x
                    ),
                    int(
                        y * scale_y
                    ),
                    int(
                        w * scale_x
                    ),
                    int(
                        h * scale_y
                    )
                )
            )

        return {
            "text": text,
            "word_count": word_count,
            "character_count": character_count,
            "boxes": scaled_boxes,
            "text_area_ratio": float(
                text_area_ratio
            ),
            "ocr_confidence": float(
                confidence
            ),
            "text_quality": float(
                quality
            ),
            "ocr_reliable": bool(
                reliable
            ),
            "ocr_variant": variant_name,
            "ocr_psm": best["psm"]
        }

    # ============================================================
    # ANALYZE VISUAL COMPLEXITY
    # ============================================================

    def analyze_visual_complexity(
        self,
        image_path,
        text_boxes
    ):

        image = cv2.imread(
            image_path
        )

        if image is None:

            return {
                "background_std": 0.0,
                "edge_density": 0.0,
                "photo_complexity": 0.0
            }

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        height, width = gray.shape

        mask = np.ones(
            gray.shape,
            dtype=np.uint8
        )

        for (
            x,
            y,
            w,
            h
        ) in text_boxes:

            x1 = max(
                0,
                x - 10
            )

            y1 = max(
                0,
                y - 10
            )

            x2 = min(
                width,
                x + w + 10
            )

            y2 = min(
                height,
                y + h + 10
            )

            mask[
                y1:y2,
                x1:x2
            ] = 0

        background_pixels = gray[
            mask == 1
        ]

        if len(
            background_pixels
        ) < 100:

            background_std = 0.0

        else:

            background_std = float(
                np.std(
                    background_pixels
                )
            )

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        background_edges = edges[
            mask == 1
        ]

        if len(
            background_edges
        ) == 0:

            edge_density = 0.0

        else:

            edge_density = float(
                np.count_nonzero(
                    background_edges
                )
                /
                len(
                    background_edges
                )
            )

        complexity_std = min(
            background_std / 80.0,
            1.0
        )

        complexity_edges = min(
            edge_density / 0.20,
            1.0
        )

        photo_complexity = (
            complexity_std
            +
            complexity_edges
        ) / 2.0

        return {
            "background_std": round(
                background_std,
                4
            ),
            "edge_density": round(
                edge_density,
                4
            ),
            "photo_complexity": round(
                photo_complexity,
                4
            )
        }

    # ============================================================
    # DETERMINE IMAGE CASE
    # ============================================================

    def detect_image_case(
        self,
        ocr_result,
        visual_analysis
    ):
        """
        Determine exactly one of:

            CASE_1_TEXT_ONLY
            CASE_2_IMAGE_ONLY
            CASE_3_TEXT_AND_IMAGE

        Decision:

            No reliable OCR
                -> IMAGE ONLY

            Reliable OCR + simple visual background
                -> TEXT ONLY

            Reliable OCR + meaningful visual content
                -> TEXT + IMAGE
        """

        if not ocr_result.get(
            "ocr_reliable",
            False
        ):

            return "image_only"

        word_count = int(
            ocr_result.get(
                "word_count",
                0
            )
        )

        character_count = int(
            ocr_result.get(
                "character_count",
                0
            )
        )

        text_quality = float(
            ocr_result.get(
                "text_quality",
                0.0
            )
        )

        ocr_confidence = float(
            ocr_result.get(
                "ocr_confidence",
                0.0
            )
        )

        text_area_ratio = float(
            ocr_result.get(
                "text_area_ratio",
                0.0
            )
        )

        photo_complexity = float(
            visual_analysis.get(
                "photo_complexity",
                0.0
            )
        )

        # --------------------------------------------------------
        # VERY STRONG TEXT-ONLY SIGNAL
        # --------------------------------------------------------

        text_only_signal = (
            text_quality >= 0.65
            and
            ocr_confidence >= 60.0
            and
            word_count >= 5
            and
            character_count >= 25
        )

        # --------------------------------------------------------
        # TEXT-ONLY IMAGE
        # --------------------------------------------------------
        #
        # Low visual complexity means the image is mainly a
        # text/document/poster-like region.
        #

        if (
            text_only_signal
            and
            photo_complexity < 0.30
        ):

            return "text_only"

        # --------------------------------------------------------
        # TEXT + IMAGE
        # --------------------------------------------------------

        if (
            word_count >= 3
            and
            character_count >= 15
            and
            text_quality >= 0.48
            and
            ocr_confidence >= 50.0
            and
            (
                photo_complexity >= 0.30
                or
                text_area_ratio < 0.10
            )
        ):

            return "text_and_image"

        # --------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------

        return "image_only"

    # ============================================================
    # NORMALIZE BINARY PROBABILITIES
    # ============================================================

    def _normalize_binary_probabilities(
        self,
        real_probability,
        ai_probability
    ):
        """
        Convert two values into a valid probability distribution.

        Output:

            real + ai = 1.0
        """

        try:

            real_probability = float(
                real_probability
            )

        except (
            ValueError,
            TypeError
        ):

            real_probability = 0.0

        try:

            ai_probability = float(
                ai_probability
            )

        except (
            ValueError,
            TypeError
        ):

            ai_probability = 0.0

        # Convert percentages to [0,1].
        if real_probability > 1.0:
            real_probability /= 100.0

        if ai_probability > 1.0:
            ai_probability /= 100.0

        # Prevent invalid negative values.
        real_probability = max(
            0.0,
            real_probability
        )

        ai_probability = max(
            0.0,
            ai_probability
        )

        total = (
            real_probability
            +
            ai_probability
        )

        if total <= 0.0:

            real_probability = 0.5
            ai_probability = 0.5

        else:

            real_probability /= total
            ai_probability /= total

        return {
            "real": float(
                real_probability
            ),
            "ai": float(
                ai_probability
            )
        }

    # ============================================================
    # RESNET50 IMAGE PREDICTION
    # ============================================================

    @torch.no_grad()
    def predict_visual(
        self,
        image_path
    ):

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        input_tensor = self.transform(
            image
        )

        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        logits = self.model(
            input_tensor
        )

        # --------------------------------------------------------
        # ACTUAL MODEL PROBABILITIES
        # --------------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        prediction = self.class_mapping[
            predicted_index
        ]

        raw_real_probability = float(
            probabilities[
                1
            ].item()
        )

        raw_ai_probability = float(
            probabilities[
                0
            ].item()
        )

        normalized = (
            self._normalize_binary_probabilities(
                raw_real_probability,
                raw_ai_probability
            )
        )

        predicted_probability = normalized[
            prediction
        ]

        return {
            "prediction": prediction,

            "confidence": float(
                predicted_probability
            ),

            "confidence_percent": round(
                predicted_probability * 100.0,
                2
            ),

            "probabilities": {
                "real": normalized["real"],
                "ai": normalized["ai"]
            },

            "probabilities_percent": {
                "real": round(
                    normalized["real"] * 100.0,
                    2
                ),
                "ai": round(
                    normalized["ai"] * 100.0,
                    2
                )
            },

            "predicted_class_probability": float(
                predicted_probability
            ),

            "model": "ResNet50",

            "model_type": "image_classifier"
        }

    # ============================================================
    # NORMALIZE ROBERTA OUTPUT
    # ============================================================

    def normalize_text_prediction(
        self,
        text_result
    ):

        if not isinstance(
            text_result,
            dict
        ):

            raise TypeError(
                "RoBERTa prediction must be a dictionary."
            )

        prediction = str(
            text_result.get(
                "prediction",
                ""
            )
        ).lower().strip()

        probabilities = text_result.get(
            "probabilities",
            {}
        )

        if not isinstance(
            probabilities,
            dict
        ):

            probabilities = {}

        # --------------------------------------------------------
        # PREDICTION NORMALIZATION
        # --------------------------------------------------------

        if prediction in [
            "label_0",
            "real",
            "real content",
            "true",
            "genuine",
            "authentic"
        ]:

            normalized_prediction = "real"

        elif prediction in [
            "label_1",
            "fake",
            "fake content",
            "ai",
            "false",
            "misleading",
            "fabricated"
        ]:

            normalized_prediction = "ai"

        else:

            normalized_prediction = prediction

        # --------------------------------------------------------
        # PROBABILITY EXTRACTION
        # --------------------------------------------------------

        real_probability = probabilities.get(
            "real",
            probabilities.get(
                "label_0",
                probabilities.get(
                    "true",
                    0.0
                )
            )
        )

        ai_probability = probabilities.get(
            "ai",
            probabilities.get(
                "fake",
                probabilities.get(
                    "label_1",
                    probabilities.get(
                        "false",
                        0.0
                    )
                )
            )
        )

        normalized_probabilities = (
            self._normalize_binary_probabilities(
                real_probability,
                ai_probability
            )
        )

        # --------------------------------------------------------
        # IF MODEL DID NOT RETURN PROBABILITIES
        # --------------------------------------------------------

        if (
            normalized_probabilities["real"] == 0.5
            and
            normalized_probabilities["ai"] == 0.5
            and
            normalized_prediction in [
                "real",
                "ai"
            ]
        ):

            confidence = text_result.get(
                "confidence",
                0.5
            )

            try:

                confidence = float(
                    confidence
                )

            except (
                ValueError,
                TypeError
            ):

                confidence = 0.5

            if confidence > 1.0:
                confidence /= 100.0

            confidence = max(
                0.0,
                min(
                    confidence,
                    1.0
                )
            )

            if normalized_prediction == "real":

                normalized_probabilities = {
                    "real": confidence,
                    "ai": 1.0 - confidence
                }

            elif normalized_prediction == "ai":

                normalized_probabilities = {
                    "real": 1.0 - confidence,
                    "ai": confidence
                }

        predicted_probability = (
            normalized_probabilities.get(
                normalized_prediction,
                0.5
            )
        )

        return {
            "prediction": normalized_prediction,

            "confidence": float(
                predicted_probability
            ),

            "confidence_percent": round(
                predicted_probability * 100.0,
                2
            ),

            "probabilities": normalized_probabilities,

            "probabilities_percent": {
                "real": round(
                    normalized_probabilities["real"]
                    * 100.0,
                    2
                ),
                "ai": round(
                    normalized_probabilities["ai"]
                    * 100.0,
                    2
                )
            },

            "model": "RoBERTa",

            "raw_result": text_result
        }

    # ============================================================
    # TEXT + IMAGE FUSION
    # ============================================================

    def fuse_predictions(
        self,
        text_prediction,
        visual_prediction
    ):
        """
        Balanced probability fusion.

        Both trained models contribute equally:

            Final REAL = (Text REAL + Image REAL) / 2
            Final AI   = (Text AI   + Image AI)   / 2

        The result is then normalized again so that:

            REAL + AI = 1.0

        Final confidence is the probability of the predicted
        class.
        """

        text_probabilities = (
            text_prediction[
                "probabilities"
            ]
        )

        visual_probabilities = (
            visual_prediction[
                "probabilities"
            ]
        )

        text_real = float(
            text_probabilities.get(
                "real",
                0.5
            )
        )

        text_ai = float(
            text_probabilities.get(
                "ai",
                0.5
            )
        )

        visual_real = float(
            visual_probabilities.get(
                "real",
                0.5
            )
        )

        visual_ai = float(
            visual_probabilities.get(
                "ai",
                0.5
            )
        )

        # --------------------------------------------------------
        # EQUAL-WEIGHT FUSION
        # --------------------------------------------------------

        combined_real = (
            text_real
            +
            visual_real
        ) / 2.0

        combined_ai = (
            text_ai
            +
            visual_ai
        ) / 2.0

        final_probabilities = (
            self._normalize_binary_probabilities(
                combined_real,
                combined_ai
            )
        )

        # --------------------------------------------------------
        # FINAL PREDICTION
        # --------------------------------------------------------

        if (
            final_probabilities["real"]
            >=
            final_probabilities["ai"]
        ):

            final_prediction = "real"

        else:

            final_prediction = "ai"

        final_confidence = (
            final_probabilities[
                final_prediction
            ]
        )

        return {
            "prediction": final_prediction,

            "confidence": float(
                final_confidence
            ),

            "confidence_percent": round(
                final_confidence * 100.0,
                2
            ),

            "probabilities": {
                "real": final_probabilities[
                    "real"
                ],
                "ai": final_probabilities[
                    "ai"
                ]
            },

            "probabilities_percent": {
                "real": round(
                    final_probabilities["real"]
                    * 100.0,
                    2
                ),
                "ai": round(
                    final_probabilities["ai"]
                    * 100.0,
                    2
                )
            },

            "fusion_method": (
                "EQUAL_WEIGHT_PROBABILITY_FUSION"
            ),

            "text_weight": 0.50,

            "visual_weight": 0.50
        }

    # ============================================================
    # BUILD COMMON IMAGE RESPONSE
    # ============================================================

    def _build_common_response(
        self,
        prediction,
        confidence,
        probabilities,
        display_prediction,
        short_prediction,
        analysis_case,
        image_type,
        ocr_result,
        visual_analysis,
        text_model_result=None,
        visual_model_result=None,
        fusion_result=None
    ):

        normalized_probabilities = (
            self._normalize_binary_probabilities(
                probabilities.get(
                    "real",
                    0.5
                ),
                probabilities.get(
                    "ai",
                    0.5
                )
            )
        )

        final_confidence = (
            normalized_probabilities[
                prediction
            ]
        )

        return {
            "status": "success",

            # ====================================================
            # FINAL RESULT
            # ====================================================

            "prediction": prediction,

            "display_prediction": display_prediction,

            "short_prediction": short_prediction,

            "confidence": float(
                final_confidence
            ),

            "confidence_percent": round(
                final_confidence * 100.0,
                2
            ),

            # ====================================================
            # PROBABILITIES
            # ====================================================

            "probabilities": {
                "real": normalized_probabilities[
                    "real"
                ],
                "ai": normalized_probabilities[
                    "ai"
                ]
            },

            "probabilities_percent": {
                "real": round(
                    normalized_probabilities["real"]
                    * 100.0,
                    2
                ),
                "ai": round(
                    normalized_probabilities["ai"]
                    * 100.0,
                    2
                )
            },

            # ====================================================
            # IMAGE MODEL
            # ====================================================

            "model": "ResNet50",

            "trained_model_name": "ResNet50",

            "visual_model_name": "ResNet50",

            "modality": "image",

            "prediction_type": (
                "image_authenticity"
            ),

            # ====================================================
            # IMAGE TYPE / CASE
            # ====================================================

            "image_type": image_type,

            "analysis_case": analysis_case,

            # ====================================================
            # OCR
            # ====================================================

            "ocr_text": ocr_result[
                "text"
            ],

            "ocr_details": ocr_result,

            # ====================================================
            # VISUAL ANALYSIS
            # ====================================================

            "visual_analysis": visual_analysis,

            # ====================================================
            # INDIVIDUAL MODELS
            # ====================================================

            "text_model_result": (
                text_model_result
            ),

            "visual_model_result": (
                visual_model_result
            ),

            # ====================================================
            # FUSION
            # ====================================================

            "fusion_result": (
                fusion_result
            ),

            # ====================================================
            # WEB VERIFICATION
            # IMAGE MUST NOT BE OVERRIDDEN BY WEB SEARCH
            # ====================================================

            "verification_enabled": False,

            "web_verification": {
                "enabled": False,
                "status": "NOT_APPLICABLE",
                "message": (
                    "Image authenticity is determined "
                    "by the trained image model."
                )
            }
        }

    # ============================================================
    # MAIN IMAGE PREDICTION
    # ============================================================

    def predict(
        self,
        image_path
    ):

        # ========================================================
        # VALIDATE INPUT
        # ========================================================

        if not isinstance(
            image_path,
            (str, os.PathLike)
        ):

            raise TypeError(
                "image_path must be a string or PathLike object."
            )

        image_path = os.fspath(
            image_path
        )

        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # ========================================================
        # STEP 1 — OCR
        # ========================================================

        ocr_result = (
            self.extract_text_from_image(
                image_path
            )
        )

        # ========================================================
        # STEP 2 — VISUAL ANALYSIS
        # ========================================================

        visual_analysis = (
            self.analyze_visual_complexity(
                image_path,
                ocr_result["boxes"]
            )
        )

        # ========================================================
        # STEP 3 — IMAGE CASE
        # ========================================================

        detected_case = (
            self.detect_image_case(
                ocr_result,
                visual_analysis
            )
        )

        # ========================================================
        # CASE 1 — TEXT ONLY
        # ========================================================

        if detected_case == "text_only":

            if not ocr_result[
                "text"
            ].strip():

                raise RuntimeError(
                    "CASE 1 was selected but OCR returned no text."
                )

            # ----------------------------------------------------
            # OCR -> ROBERTA
            # ----------------------------------------------------

            text_result = (
                self.text_engine.predict(
                    ocr_result["text"]
                )
            )

            normalized_text = (
                self.normalize_text_prediction(
                    text_result
                )
            )

            prediction = normalized_text[
                "prediction"
            ]

            probabilities = (
                normalized_text[
                    "probabilities"
                ]
            )

            return self._build_common_response(

                prediction=prediction,

                confidence=normalized_text[
                    "confidence"
                ],

                probabilities=probabilities,

                display_prediction=(
                    "FAKE CONTENT"
                    if prediction == "ai"
                    else "REAL CONTENT"
                ),

                short_prediction=(
                    "FAKE"
                    if prediction == "ai"
                    else "REAL"
                ),

                analysis_case=(
                    "CASE_1_TEXT_ONLY"
                ),

                image_type="text",

                ocr_result=ocr_result,

                visual_analysis=visual_analysis,

                text_model_result=(
                    normalized_text
                ),

                visual_model_result=None,

                fusion_result=None
            )

        # ========================================================
        # CASE 2 — IMAGE ONLY
        # ========================================================

        if detected_case == "image_only":

            # ----------------------------------------------------
            # IMAGE -> RESNET50
            # ----------------------------------------------------

            visual_result = (
                self.predict_visual(
                    image_path
                )
            )

            prediction = visual_result[
                "prediction"
            ]

            probabilities = (
                visual_result[
                    "probabilities"
                ]
            )

            return self._build_common_response(

                prediction=prediction,

                confidence=visual_result[
                    "confidence"
                ],

                probabilities=probabilities,

                display_prediction=(
                    self.display_mapping[
                        prediction
                    ]
                ),

                short_prediction=(
                    "AI GENERATED"
                    if prediction == "ai"
                    else "REAL"
                ),

                analysis_case=(
                    "CASE_2_IMAGE_ONLY"
                ),

                image_type="image",

                ocr_result=ocr_result,

                visual_analysis=visual_analysis,

                text_model_result=None,

                visual_model_result=(
                    visual_result
                ),

                fusion_result=None
            )

        # ========================================================
        # CASE 3 — TEXT + IMAGE
        # ========================================================

        if detected_case == "text_and_image":

            if not ocr_result[
                "text"
            ].strip():

                # Safety fallback.
                visual_result = (
                    self.predict_visual(
                        image_path
                    )
                )

                prediction = visual_result[
                    "prediction"
                ]

                return self._build_common_response(

                    prediction=prediction,

                    confidence=visual_result[
                        "confidence"
                    ],

                    probabilities=visual_result[
                        "probabilities"
                    ],

                    display_prediction=(
                        self.display_mapping[
                            prediction
                        ]
                    ),

                    short_prediction=(
                        "AI GENERATED"
                        if prediction == "ai"
                        else "REAL"
                    ),

                    analysis_case=(
                        "CASE_2_IMAGE_ONLY"
                    ),

                    image_type="image",

                    ocr_result=ocr_result,

                    visual_analysis=visual_analysis,

                    text_model_result=None,

                    visual_model_result=(
                        visual_result
                    ),

                    fusion_result=None
                )

            # ----------------------------------------------------
            # OCR -> ROBERTA
            # ----------------------------------------------------

            text_result = (
                self.text_engine.predict(
                    ocr_result["text"]
                )
            )

            normalized_text = (
                self.normalize_text_prediction(
                    text_result
                )
            )

            # ----------------------------------------------------
            # IMAGE -> RESNET50
            # ----------------------------------------------------

            visual_result = (
                self.predict_visual(
                    image_path
                )
            )

            # ----------------------------------------------------
            # BALANCED FUSION
            # ----------------------------------------------------

            fusion_result = (
                self.fuse_predictions(
                    normalized_text,
                    visual_result
                )
            )

            prediction = fusion_result[
                "prediction"
            ]

            return self._build_common_response(

                prediction=prediction,

                confidence=fusion_result[
                    "confidence"
                ],

                probabilities=fusion_result[
                    "probabilities"
                ],

                display_prediction=(
                    "AI-GENERATED IMAGE"
                    if prediction == "ai"
                    else "REAL IMAGE"
                ),

                short_prediction=(
                    "AI GENERATED"
                    if prediction == "ai"
                    else "REAL"
                ),

                analysis_case=(
                    "CASE_3_TEXT_AND_IMAGE"
                ),

                image_type="text+image",

                ocr_result=ocr_result,

                visual_analysis=visual_analysis,

                text_model_result=(
                    normalized_text
                ),

                visual_model_result=(
                    visual_result
                ),

                fusion_result=fusion_result
            )

        # ========================================================
        # UNEXPECTED CASE
        # ========================================================

        raise RuntimeError(
            "Unexpected image analysis case:\n"
            f"{detected_case}"
        )