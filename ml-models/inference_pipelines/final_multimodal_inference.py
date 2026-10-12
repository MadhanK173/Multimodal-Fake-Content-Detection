
import os
import sys
import json


PROJECT_ROOT = "/content/drive/MyDrive/Multimodal_Fake_Content_Detection"

DEPLOYMENT_ROOT = os.path.join(
    PROJECT_ROOT,
    "deployment"
)

PIPELINE_DIR = os.path.join(
    DEPLOYMENT_ROOT,
    "inference_pipelines"
)

CONFIG_DIR = os.path.join(
    DEPLOYMENT_ROOT,
    "inference_config"
)


if PIPELINE_DIR not in sys.path:

    sys.path.insert(
        0,
        PIPELINE_DIR
    )


from unified_multimodal_inference import (
    UnifiedMultimodalInferenceEngine
)


LABEL_CONFIG_PATH = os.path.join(
    CONFIG_DIR,
    "deployment_label_mapping.json"
)


with open(
    LABEL_CONFIG_PATH,
    "r"
) as f:

    LABEL_CONFIG = json.load(f)


LABEL_MAPPING = LABEL_CONFIG["mappings"]


def standardize_prediction(
    modality,
    raw_result
):


    result = raw_result.get(
        "result",
        raw_result
    )


    # ============================================================
    # TEXT
    # ============================================================

    if modality == "text":


        prediction = result.get(
            "prediction"
        )


        mapping = LABEL_MAPPING[
            "text"
        ].get(
            prediction
        )


        if mapping is None:

            return {

                "status": "error",

                "modality": "text",

                "message":

                    f"Unknown prediction label: {prediction}"

            }


        return {

            "status": "success",

            "modality": "text",

            "prediction":
                mapping["display_label"],

            "short_prediction":
                mapping["short_label"],

            "class_id":
                mapping["class_id"],

            "confidence":
                result.get("confidence"),

            "confidence_percent":
                result.get("confidence_percent"),

            "probabilities":
                result.get("probabilities"),

            "model":
                result.get(
                    "model",
                    "RoBERTa"
                ),

            "prediction_type":
                "content_authenticity"

        }


    # ============================================================
    # IMAGE
    # ============================================================

    elif modality == "image":


        prediction = result.get(
            "prediction"
        )


        mapping = LABEL_MAPPING[
            "image"
        ].get(
            prediction
        )


        if mapping is None:

            return {

                "status": "error",

                "modality": "image",

                "message":

                    f"Unknown prediction label: {prediction}"

            }


        final_result = {

            "status": "success",

            "modality": "image",

            "prediction":
                mapping["display_label"],

            "short_prediction":
                mapping["short_label"],

            "class_id":
                mapping["class_id"],

            "confidence":
                result.get("confidence"),

            "confidence_percent":
                result.get("confidence_percent"),

            "probabilities":
                result.get("probabilities"),

            "model":
                result.get(
                    "model",
                    "Image Model"
                ),

            "prediction_type":
                "image_authenticity"

        }


        # Additional multimodal image information

        if "analysis_case" in result:

            final_result[
                "analysis_case"
            ] = result[
                "analysis_case"
            ]


        if "fusion_method" in result:

            final_result[
                "fusion_method"
            ] = result[
                "fusion_method"
            ]


        if "ocr_text" in result:

            final_result[
                "ocr_text"
            ] = result[
                "ocr_text"
            ]


        if "text_model_result" in result:

            final_result[
                "text_model_result"
            ] = result[
                "text_model_result"
            ]


        if "visual_model_result" in result:

            final_result[
                "visual_model_result"
            ] = result[
                "visual_model_result"
            ]


        return final_result


    # ============================================================
    # AUDIO
    # ============================================================

    elif modality == "audio":


        classification = result.get(
            "classification",
            {}
        )


        prediction = classification.get(
            "prediction"
        )


        mapping = LABEL_MAPPING[
            "audio"
        ].get(
            prediction
        )


        if mapping is None:

            return {

                "status": "error",

                "modality": "audio",

                "message":

                    f"Unknown prediction label: {prediction}"

            }


        return {

            "status": "success",

            "modality": "audio",

            "prediction":
                mapping["display_label"],

            "short_prediction":
                mapping["short_label"],

            "class_id":
                mapping["class_id"],

            "confidence":
                classification.get(
                    "confidence"
                ),

            "confidence_percent":
                classification.get(
                    "confidence_percent"
                ),

            "probabilities":
                classification.get(
                    "probabilities"
                ),

            "model":
                classification.get(
                    "model",
                    "RoBERTa"
                ),

            "transcription_model":
                "Whisper",

            "language":
                result.get("language"),

            "transcript":
                result.get("transcript"),

            "prediction_type":
                "content_authenticity"

        }


    else:

        return {

            "status": "error",

            "message":

                f"Unsupported modality: {modality}"

        }


class FinalMultimodalInferenceEngine:


    def __init__(
        self,
        device=None
    ):


        self.base_engine = UnifiedMultimodalInferenceEngine(

            device=device

        )


    def predict(
        self,
        input_data
    ):


        raw_output = self.base_engine.predict(

            input_data

        )


        modality = raw_output.get(
            "modality"
        )


        return standardize_prediction(

            modality,

            raw_output

        )
