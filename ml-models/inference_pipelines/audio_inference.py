import os

import sys

from pathlib import Path



import torch

import whisper





class AudioInferenceEngine:

    """

    LOCAL WINDOWS AUDIO INFERENCE ENGINE



    Architecture:



        Audio File

             ↓

        Whisper Speech-to-Text

             ↓

        Transcript

             ↓

        RoBERTa Text Classification

             ↓

        REAL CONTENT / FAKE CONTENT

    """



    def __init__(

        self,

        whisper_model_name="base",

        device=None

    ):

        # ============================================================

        # LOCAL PROJECT ROOT

        # ============================================================



        current_dir = Path(

            __file__

        ).resolve()



        self.project_root = current_dir.parents[2]



        # ============================================================

        # LOCAL PIPELINE DIRECTORY

        # ============================================================



        self.pipeline_dir = (

            self.project_root

            / "ml-models"

            / "inference_pipelines"

        )



        # ============================================================

        # LOCAL ROBERTA MODEL

        # ============================================================



        self.text_model_path = (

            self.project_root

            / "ml-models"

            / "text"

            / "roberta_base_onnx"

        )



        if not self.pipeline_dir.is_dir():

            raise FileNotFoundError(

                "Inference pipeline directory not found:\n"

                f"{self.pipeline_dir}"

            )



        if not self.text_model_path.is_dir():

            raise FileNotFoundError(

                "RoBERTa model directory not found:\n"

                f"{self.text_model_path}"

            )



        pipeline_dir_string = str(

            self.pipeline_dir

        )



        if pipeline_dir_string not in sys.path:

            sys.path.insert(

                0,

                pipeline_dir_string

            )



        # ============================================================

        # CONFIGURE FFMPEG FOR WINDOWS

        # ============================================================



        self._configure_ffmpeg()



        # ============================================================

        # DEVICE

        # ============================================================



        if device is None:

            self.device = (

                "cuda"

                if torch.cuda.is_available()

                else "cpu"

            )

        else:

            self.device = str(device)



        print(

            f"Audio inference device: {self.device}"

        )



        # ============================================================

        # LOAD WHISPER

        # ============================================================



        print(

            f"Loading Whisper model: {whisper_model_name}"

        )



        try:

            self.whisper_model = whisper.load_model(

                whisper_model_name,

                device=self.device

            )

        except Exception as exc:

            raise RuntimeError(

                "Failed to load Whisper model.\n"

                f"Model: {whisper_model_name}\n"

                f"Device: {self.device}\n"

                f"Error: {exc}"

            ) from exc



        print(

            "Whisper loaded successfully."

        )



        # ============================================================

        # LOAD LOCAL ROBERTA PIPELINE

        # ============================================================



        print(

            "Loading local RoBERTa text pipeline..."

        )



        try:

            from text_inference import (

                TextInferenceEngine

            )

        except Exception as exc:

            raise RuntimeError(

                "Unable to import local text pipeline.\n"

                f"Pipeline: {self.pipeline_dir}\n"

                f"Error: {exc}"

            ) from exc



        self.text_engine = TextInferenceEngine(

            model_path=str(

                self.text_model_path

            ),

            max_length=512,

            device=self.device

        )



        print(

            "RoBERTa TextInferenceEngine loaded successfully."

        )



        print(

            "AudioInferenceEngine initialized successfully."

        )



    # ================================================================

    # FFMPEG CONFIGURATION

    # ================================================================



    def _configure_ffmpeg(self):

        """

        Configure FFmpeg explicitly for Windows.



        This avoids depending on a temporary PowerShell PATH.

        """



        possible_ffmpeg_paths = [

            Path(

                r"C:\Program Files\UiPath\Studio\ffmpeg\ffmpeg.exe"

            ),

            Path(

                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

            ),

            Path(

                r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"

            )

        ]



        for ffmpeg_path in possible_ffmpeg_paths:



            if ffmpeg_path.is_file():



                ffmpeg_dir = str(

                    ffmpeg_path.parent

                )



                os.environ["PATH"] = (

                    ffmpeg_dir

                    + os.pathsep

                    + os.environ.get(

                        "PATH",

                        ""

                    )

                )



                self.ffmpeg_path = str(

                    ffmpeg_path

                )



                print(

                    "FFmpeg configured:"

                )



                print(

                    self.ffmpeg_path

                )



                return



        # Fallback: check whether FFmpeg is already on PATH.

        ffmpeg_found = False



        for path_entry in os.environ.get(

            "PATH",

            ""

        ).split(

            os.pathsep

        ):



            candidate = Path(

                path_entry

            ) / "ffmpeg.exe"



            if candidate.is_file():



                self.ffmpeg_path = str(

                    candidate

                )



                ffmpeg_found = True

                break



        if not ffmpeg_found:



            raise RuntimeError(

                "FFmpeg executable could not be found. "

                "Expected location includes:\n"

                r"C:\Program Files\UiPath\Studio\ffmpeg\ffmpeg.exe"

            )



        print(

            "FFmpeg found on PATH:"

        )



        print(

            self.ffmpeg_path

        )



    # ================================================================

    # TRANSCRIBE AUDIO

    # ================================================================



    def transcribe_audio(

        self,

        audio_path

    ):

        if not isinstance(

            audio_path,

            (str, os.PathLike)

        ):

            raise TypeError(

                "audio_path must be a string or PathLike object."

            )



        audio_path = os.path.abspath(

            os.fspath(

                audio_path

            )

        )



        if not os.path.isfile(

            audio_path

        ):

            raise FileNotFoundError(

                f"Audio file not found:\n{audio_path}"

            )



        print(

            f"Transcribing audio:\n{audio_path}"

        )



        try:



            result = self.whisper_model.transcribe(

                audio_path,

                fp16=False

            )



        except Exception as exc:



            raise RuntimeError(

                "Audio transcription failed.\n"

                f"Audio file: {audio_path}\n"

                f"FFmpeg: {getattr(self, 'ffmpeg_path', 'unknown')}\n"

                f"Error: {exc}"

            ) from exc



        transcript = str(

            result.get(

                "text",

                ""

            )

        ).strip()



        return {

            "transcript": transcript,

            "language": result.get(

                "language"

            )

        }



    # ================================================================

    # CLASSIFY TRANSCRIPT

    # ================================================================



    def classify_transcript(

        self,

        transcript

    ):



        if not isinstance(

            transcript,

            str

        ):

            raise TypeError(

                "Transcript must be a string."

            )



        transcript = transcript.strip()



        if not transcript:

            raise ValueError(

                "Transcript cannot be empty."

            )



        return self.text_engine.predict(

            transcript

        )



    # ================================================================

    # COMPLETE AUDIO PREDICTION

    # ================================================================



    def predict(

        self,

        audio_path

    ):



        audio_path = os.path.abspath(

            os.fspath(

                audio_path

            )

        )



        transcription_result = (

            self.transcribe_audio(

                audio_path

            )

        )



        transcript = transcription_result.get(

            "transcript",

            ""

        ).strip()



        language = transcription_result.get(

            "language"

        )



        if not transcript:



            return {

                "status": "failed",

                "modality": "audio",

                "prediction_type": "content_authenticity",

                "audio_path": audio_path,

                "transcript": "",

                "language": language,

                "error": "No speech transcript extracted"

            }



        classification_result = (

            self.classify_transcript(

                transcript

            )

        )



        return {

            "status": "success",

            "modality": "audio",

            "prediction_type": "content_authenticity",

            "audio_path": audio_path,

            "transcript": transcript,

            "language": language,

            "classification": classification_result

        }