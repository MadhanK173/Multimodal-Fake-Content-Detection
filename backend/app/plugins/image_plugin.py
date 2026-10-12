from typing import Any, Dict

from ..services.multimodal_service import predict_image


class ImagePlugin:
    """
    Plugin responsible for image-based fake content detection.

    Flow:
    Image
      ↓
    Existing multimodal image pipeline
      ↓
    OCR / ResNet50 / multimodal fusion
      ↓
    Structured plugin result
    """

    name = "image"
    modality = "image"

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image using the existing trained
        multimodal image inference pipeline.
        """

        if not image_path:
            raise ValueError("Image path cannot be empty.")

        # Use the existing image inference system.
        model_result = predict_image(image_path)

        return {
            "plugin": self.name,
            "modality": self.modality,
            "model_result": model_result,
            "processing_pipeline": [
                "Image Input",
                "OCR",
                "Visual Analysis",
                "ResNet50",
                "Multimodal Fusion",
            ],
            "status": "success",
        }