from typing import Any, Dict


class ResultPlugin:
    """
    Plugin responsible for creating a consistent result
    structure from model and verification outputs.

    This plugin does not change the trained model prediction.
    """

    name = "result"
    modality = "result"

    def build_result(
        self,
        model_result: Dict[str, Any],
        web_verification: Dict[str, Any] | None = None,
        *,
        input_text: str = "",
        transcript: str = "",
    ) -> Dict[str, Any]:
        """
        Build the common result structure returned by
        the plugin layer.
        """

        if not isinstance(model_result, dict):
            raise ValueError(
                "model_result must be a dictionary."
            )

        if web_verification is None:
            web_verification = {}

        if not isinstance(web_verification, dict):
            raise ValueError(
                "web_verification must be a dictionary."
            )

        result = {
            "plugin": self.name,
            "status": "success",
            "model_result": model_result,
            "web_verification": web_verification,
        }

        if input_text:
            result["input_text"] = input_text

        if transcript:
            result["transcript"] = transcript

        return result