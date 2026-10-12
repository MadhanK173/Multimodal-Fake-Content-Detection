from typing import Any, Dict

from ..services.web_verification_service import WebVerificationService


class EvidencePlugin:
    """
    Plugin responsible for web-based evidence verification.

    This plugin uses the existing WebVerificationService,
    which performs the Tavily search and evidence analysis.
    """

    name = "evidence"
    modality = "evidence"

    def __init__(self):
        self.web_verifier = WebVerificationService()

    def verify(
        self,
        text: str,
        model_result: Dict[str, Any],
        modality: str = "text",
    ) -> Dict[str, Any]:
        """
        Verify a claim using the existing Tavily-based
        web verification service.
        """

        text = (text or "").strip()

        if not text:
            raise ValueError(
                "Text cannot be empty for evidence verification."
            )

        if not isinstance(model_result, dict):
            raise ValueError(
                "model_result must be a dictionary."
            )

        result = self.web_verifier.verify_text(
            text=text,
            model_result=model_result,
            modality=modality,
        )

        return {
            "plugin": self.name,
            "modality": modality,
            "verification": result,
            "status": "success",
        }