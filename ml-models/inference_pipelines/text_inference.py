import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TextInferenceEngine:

    def __init__(
        self,
        model_path=None,
        max_length=512,
        device=None
    ):

        # Automatically locate project root
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
                "roberta_base_final_RECOVERED"
            )


        # Normalize path
        self.model_path = os.path.abspath(
            model_path
        )

        self.max_length = max_length


        # Validate model directory
        if not os.path.exists(
            self.model_path
        ):

            raise FileNotFoundError(
                f"Text model directory not found:\n"
                f"{self.model_path}"
            )


        print(
            f"Loading RoBERTa model from:\n"
            f"{self.model_path}"
        )


        # Device selection
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


        print(
            f"Using device: {self.device}"
        )


        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )


        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True
        )


        self.model.to(
            self.device
        )

        self.model.eval()


        # Label mapping
        self.id_to_label = {

            idx: str(
                self.model.config.id2label.get(
                    idx,
                    str(idx)
                )
            ).lower()

            for idx in range(
                self.model.config.num_labels
            )

        }


        print(
            "RoBERTa model loaded successfully!"
        )


    @torch.no_grad()
    def predict(self, text):

        if not isinstance(text, str):

            raise TypeError(
                "Input must be a string."
            )


        text = text.strip()


        if not text:

            raise ValueError(
                "Input text cannot be empty."
            )


        # Tokenize
        inputs = self.tokenizer(

            text,

            max_length=self.max_length,

            truncation=True,

            padding=True,

            return_tensors="pt"

        )


        inputs = {

            key: value.to(
                self.device
            )

            for key, value
            in inputs.items()

        }


        # Inference
        outputs = self.model(
            **inputs
        )


        probabilities = torch.softmax(

            outputs.logits,

            dim=1

        )[0]


        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )


        confidence = float(
            probabilities[
                predicted_index
            ].item()
        )


        probability_dict = {

            self.id_to_label.get(
                index,
                str(index)
            ): float(
                probability.item()
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