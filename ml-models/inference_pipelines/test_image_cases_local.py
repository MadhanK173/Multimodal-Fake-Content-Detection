import json
import os

from image_inference import ImageInferenceEngine


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

IMAGE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ml-models",
    "image",
    "resnet50_best.pth"
)


def print_result(title, result):

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )


def test_image(engine, title, image_path):

    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)

    print("Image:")
    print(image_path)

    print("\nExists:")
    print(os.path.exists(image_path))

    if not os.path.exists(image_path):
        print("❌ FILE NOT FOUND")
        return None

    try:

        result = engine.predict(
            image_path
        )

        print_result(
            title,
            result
        )

        return result

    except Exception as exc:

        print("\n❌ TEST FAILED")
        print(
            f"Exception Type: {type(exc).__name__}"
        )

        print(
            f"Exception Message: {exc}"
        )

        return None


def main():

    print("=" * 90)
    print("LOCAL THREE-CASE IMAGE VALIDATION")
    print("=" * 90)

    print("\nInitializing ImageInferenceEngine...")

    engine = ImageInferenceEngine(
        model_path=IMAGE_MODEL_PATH
    )

    print("\n✅ ImageInferenceEngine initialized.")

    print("\n")
    print("IMPORTANT:")
    print("Provide one image for each case.")
    print()
    print("CASE 1 = text-only image")
    print("CASE 2 = image-only photograph")
    print("CASE 3 = text + photograph")

    # -----------------------------------------------------------------
    # CHANGE THESE THREE PATHS
    # -----------------------------------------------------------------

    case1_path = os.path.join(
        PROJECT_ROOT,
        "test-data",
        "case1_text_only.jpg"
    )

    case2_path = os.path.join(
        PROJECT_ROOT,
        "test-data",
        "case2_image_only.jpg"
    )

    case3_path = os.path.join(
        PROJECT_ROOT,
        "test-data",
        "case3_text_and_image.jpg"
    )

    # -----------------------------------------------------------------
    # RUN TESTS
    # -----------------------------------------------------------------

    case1_result = test_image(
        engine,
        "CASE 1 — TEXT ONLY IMAGE",
        case1_path
    )

    case2_result = test_image(
        engine,
        "CASE 2 — IMAGE ONLY",
        case2_path
    )

    case3_result = test_image(
        engine,
        "CASE 3 — TEXT + IMAGE",
        case3_path
    )

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL THREE-CASE VALIDATION SUMMARY")
    print("=" * 90)

    results = {
        "CASE_1_TEXT_ONLY": case1_result,
        "CASE_2_IMAGE_ONLY": case2_result,
        "CASE_3_TEXT_AND_IMAGE": case3_result
    }

    for case_name, result in results.items():

        if result is None:

            print(f"{case_name}: ❌ TEST FAILED")

            continue

        detected_case = result.get(
            "analysis_case"
        )

        print(
            f"{case_name}: "
            f"Detected = {detected_case}"
        )

        if (
            detected_case == case_name
        ):

            print("   ✅ PASS")

        else:

            print("   ❌ FAIL")

    print("\n" + "=" * 90)
    print("VALIDATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()