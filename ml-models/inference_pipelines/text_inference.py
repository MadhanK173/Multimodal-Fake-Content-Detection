import os
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class TextInferenceEngine:

    def __init__(
        self,
        model_path=None,
        max_length=512,
        device=None
    ):

        # ---------------------------------------------------------
        # Locate project root
        #
        # Current file:
        # project/
        #   ml-models/
        #     inference_pipelines/
        #       text_inference.py
        #
        # Therefore project root = two levels above this file.
        # ---------------------------------------------------------
        if model_path is None:

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

            model_path = os.path.join(
                project_root,
                "ml-models",
                "text",
                "roberta_base_onnx"
            )

        self.model_path = os.path.abspath(
            model_path
        )

        self.max_length = max_length

        # ---------------------------------------------------------
        # Validate model directory
        # ---------------------------------------------------------
        if not os.path.exists(self.model_path):

            raise FileNotFoundError(
                f"ONNX text model directory not found:\n"
                f"{self.model_path}"
            )

        self.onnx_model_path = os.path.join(
            self.model_path,
            "model_int8.onnx"
        )

        if not os.path.exists(
            self.onnx_model_path
        ):

            raise FileNotFoundError(
                f"INT8 ONNX model not found:\n"
                f"{self.onnx_model_path}"
            )

        print(
            "Loading INT8 RoBERTa ONNX model from:"
        )

        print(
            self.onnx_model_path
        )

        # ---------------------------------------------------------
        # CPU execution
        # ---------------------------------------------------------
        self.device = "cpu"

        print(
            "Using device: cpu (ONNX Runtime)"
        )

        # ---------------------------------------------------------
        # Load tokenizer
        # ---------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        # ---------------------------------------------------------
        # ONNX Runtime session
        #
        # Keep thread count low for Render Free.
        # ---------------------------------------------------------
        session_options = ort.SessionOptions()

        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=session_options,
            providers=[
                "CPUExecutionProvider"
            ]
        )

        # ---------------------------------------------------------
        # Get model input/output names
        # ---------------------------------------------------------
        self.input_names = {
            input_info.name
            for input_info
            in self.session.get_inputs()
        }

        self.output_names = [
            output_info.name
            for output_info
            in self.session.get_outputs()
        ]

        # ---------------------------------------------------------
        # Fixed mapping from the trained model
        #
        # Class 0 = REAL
        # Class 1 = FAKE
        # ---------------------------------------------------------
        self.id_to_label = {
            0: "real",
            1: "fake"
        }

        print(
            "RoBERTa INT8 ONNX model loaded successfully!"
        )

        print(
            f"ONNX inputs: {sorted(self.input_names)}"
        )

        print(
            f"ONNX outputs: {self.output_names}"
        )

    def predict(
        self,
        text: str
    ) -> Dict[str, Any]:

        # ---------------------------------------------------------
        # Validate input
        # ---------------------------------------------------------
        if not isinstance(text, str):

            raise TypeError(
                "Input must be a string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "Input text cannot be empty."
            )

        # ---------------------------------------------------------
        # Tokenize
        # ---------------------------------------------------------
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="np"
        )

        # ---------------------------------------------------------
        # Prepare ONNX inputs
        # ---------------------------------------------------------
        inputs = {}

        for name in self.input_names:

            if name not in encoded:

                raise RuntimeError(
                    f"Required ONNX input '{name}' "
                    f"was not produced by the tokenizer."
                )

            inputs[name] = encoded[
                name
            ].astype(
                np.int64
            )

        # ---------------------------------------------------------
        # Inference
        # ---------------------------------------------------------
        outputs = self.session.run(
            self.output_names,
            inputs
        )

        logits = np.asarray(
            outputs[0],
            dtype=np.float32
        )[0]

        # ---------------------------------------------------------
        # Softmax
        # ---------------------------------------------------------
        shifted_logits = (
            logits - np.max(logits)
        )

        exp_logits = np.exp(
            shifted_logits
        )

        probabilities = (
            exp_logits /
            np.sum(exp_logits)
        )

        # ---------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------
        predicted_index = int(
            np.argmax(
                probabilities
            )
        )

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        probability_dict = {

            self.id_to_label.get(
                index,
                str(index)
            ): float(
                probability
            )

            for index, probability
            in enumerate(
                probabilities
            )
        }

        prediction = self.id_to_label.get(
            predicted_index,
            str(predicted_index)
        )

        # ---------------------------------------------------------
        # Preserve existing API response format
        # ---------------------------------------------------------
        return {

            "prediction": prediction,

            "confidence": confidence,

            "confidence_percent": round(
                confidence * 100,
                2
            ),

            "probabilities": {

                label: round(
                    probability * 100,
                    4
                )

                for label, probability
                in probability_dict.items()

            },

            "model": "RoBERTa",

            "modality": "text"
        }