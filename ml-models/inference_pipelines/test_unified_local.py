import json
import os

from unified_multimodal_inference import (
    UnifiedMultimodalInferenceEngine
)


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

TEXT_SAMPLE = (
    "Artificial intelligence is transforming industries "
    "by automating tasks and improving decision-making."
)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "test-data",
    "case3_text_and_image.jpg"
)

AUDIO_PATH = os.path.join(
    PROJECT_ROOT,
    "test-data",
    "test_audio.mp3"
)


def show_result(
    title,
    result
):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )


def main():

    print("=" * 80)
    print("UNIFIED MULTIMODAL LOCAL TEST")
    print("=" * 80)

    print("\nInitializing unified engine...")

    engine = UnifiedMultimodalInferenceEngine()

    print(
        "\n✅ UnifiedMultimodalInferenceEngine initialized."
    )

    # ============================================================
    # TEXT
    # ============================================================

    print(
        "\nRunning TEXT test..."
    )

    text_result = engine.predict(
        TEXT_SAMPLE
    )

    show_result(
        "TEXT RESULT",
        text_result
    )

    # ============================================================
    # IMAGE
    # ============================================================

    print(
        "\nRunning IMAGE test..."
    )

    print(
        "Image exists:",
        os.path.isfile(IMAGE_PATH)
    )

    if os.path.isfile(IMAGE_PATH):

        image_result = engine.predict(
            IMAGE_PATH
        )

        show_result(
            "IMAGE RESULT",
            image_result
        )

    else:

        print(
            "⚠️ Image test file not found:"
        )

        print(
            IMAGE_PATH
        )

    # ============================================================
    # AUDIO
    # ============================================================

    print(
        "\nRunning AUDIO test..."
    )

    print(
        "Audio exists:",
        os.path.isfile(AUDIO_PATH)
    )

    if os.path.isfile(AUDIO_PATH):

        audio_result = engine.predict(
            AUDIO_PATH
        )

        show_result(
            "AUDIO RESULT",
            audio_result
        )

    else:

        print(
            "⚠️ Audio test file not found:"
        )

        print(
            AUDIO_PATH
        )

    # ============================================================
    # EXPLICIT MODALITY TEST
    # ============================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "AUTOMATIC MODALITY DETECTION TEST"
    )

    print(
        "=" * 80
    )

    print(
        "Text:",
        engine.detect_modality(
            TEXT_SAMPLE
        )
    )

    if os.path.isfile(IMAGE_PATH):
        print(
            "Image:",
            engine.detect_modality(
                IMAGE_PATH
            )
        )

    if os.path.isfile(AUDIO_PATH):
        print(
            "Audio:",
            engine.detect_modality(
                AUDIO_PATH
            )
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "UNIFIED MULTIMODAL TEST COMPLETED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()