import json
import os

from audio_inference import AudioInferenceEngine


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

AUDIO_PATH = os.path.join(
    PROJECT_ROOT,
    "test-data",
    "test_audio.mp3"
)


def main():
    print("=" * 80)
    print("LOCAL AUDIO INFERENCE TEST")
    print("=" * 80)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nAudio file:")
    print(AUDIO_PATH)

    print("\nAudio file exists:")
    print(os.path.isfile(AUDIO_PATH))

    if not os.path.isfile(AUDIO_PATH):
        print("\n❌ Audio test file not found.")
        return

    print("\nInitializing AudioInferenceEngine...")

    engine = AudioInferenceEngine(
        whisper_model_name="base"
    )

    print(
        "\n✅ AudioInferenceEngine initialized successfully!"
    )

    print("\nRunning audio inference...")

    result = engine.predict(
        AUDIO_PATH
    )

    print("\n" + "=" * 80)
    print("AUDIO TEST RESULT")
    print("=" * 80)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\n" + "=" * 80)
    print("AUDIO TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()