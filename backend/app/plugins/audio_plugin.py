from typing import Any, Dict

from ..services.multimodal_service import predict_audio
from ..services.web_verification_service import WebVerificationService


class AudioPlugin:
    """
    Plugin responsible for audio-based fake content detection.

    Flow:
    Audio
      ↓
    Whisper
      ↓
    Transcript
      ↓
    RoBERTa
      ↓
    Tavily web verification using transcript only
    """

    name = "audio"
    modality = "audio"

    def __init__(self):
        self.web_verifier = WebVerificationService()

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze an audio file using the existing
        Whisper + RoBERTa pipeline and Tavily verification.
        """

        if not audio_path:
            raise ValueError("Audio path cannot be empty.")

        # Step 1: Existing Whisper + RoBERTa pipeline
        model_result = predict_audio(audio_path)

        # Step 2: Extract the Whisper transcript
        transcript = self._extract_transcript(model_result)

        if not transcript:
            raise ValueError(
                "No usable transcript was extracted from the audio."
            )

        # Step 3: Verify ONLY the transcript using Tavily
        web_result = self.web_verifier.verify_text(
            text=transcript,
            model_result=model_result,
            modality="audio_transcript",
        )

        # Step 4: Return unified plugin result
        return {
            "plugin": self.name,
            "modality": self.modality,
            "model_result": model_result,
            "transcript": transcript,
            "web_verification": web_result,
            "verification_enabled": bool(
                web_result.get("enabled", False)
            ),
            "processing_pipeline": [
                "Audio Input",
                "Whisper Speech-to-Text",
                "RoBERTa",
                "Live Tavily Web Search",
                "Evidence Verification",
            ],
            "status": "success",
        }

    @staticmethod
    def _extract_transcript(result: Any) -> str:
        """
        Extract transcript from the existing audio
        inference result.

        Supports common result layouts used by
        the existing inference pipeline.
        """

        if not isinstance(result, dict):
            return ""

        # Direct transcript
        transcript = result.get("transcript")

        if isinstance(transcript, str) and transcript.strip():
            return transcript.strip()

        # Nested result
        nested_result = result.get("result")

        if isinstance(nested_result, dict):
            transcript = nested_result.get("transcript")

            if isinstance(transcript, str) and transcript.strip():
                return transcript.strip()

        # Nested transcription field
        transcription = result.get("transcription")

        if isinstance(transcription, str) and transcription.strip():
            return transcription.strip()

        if isinstance(transcription, dict):
            text = transcription.get("text")

            if isinstance(text, str) and text.strip():
                return text.strip()

        return ""