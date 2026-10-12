from typing import Any, Dict

from ..services.multimodal_service import predict_text
from ..services.web_verification_service import WebVerificationService


class TextPlugin:
    """
    Plugin responsible for text-based fake content detection.

    Flow:
    Text
      ↓
    Existing RoBERTa model
      ↓
    Tavily web verification
      ↓
    Structured plugin result
    """

    name = "text"
    modality = "text"

    def __init__(self):
        self.web_verifier = WebVerificationService()

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze a text input using the existing
        text ML model and web verification service.
        """

        text = (text or "").strip()

        if not text:
            raise ValueError("Text input cannot be empty.")

        # Step 1: Existing RoBERTa-based prediction
        model_result = predict_text(text)

        # Step 2: Existing Tavily web verification
        web_result = self.web_verifier.verify_text(
            text=text,
            model_result=model_result,
            modality="text",
        )

        # Step 3: Return a unified plugin result
        return {
            "plugin": self.name,
            "modality": self.modality,
            "input_text": text,
            "model_result": model_result,
            "web_verification": web_result,
            "verification_enabled": bool(
                web_result.get("enabled", False)
            ),
            "processing_pipeline": [
                "Text Input",
                "RoBERTa",
                "Live Tavily Web Search",
                "Evidence Verification",
            ],
            "status": "success",
        }