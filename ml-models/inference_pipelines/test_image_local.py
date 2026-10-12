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

IMAGE_MODEL = os.path.join(
    PROJECT_ROOT,
    "ml-models",
    "image",
    "resnet50_best.pth"
)


def print_result(title, result):
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
    print("LOCAL IMAGE INFERENCE TEST")
    print("=" * 80)

    print("\nResNet50:")
    print(IMAGE_MODEL)
    print("Exists:", os.path.exists(IMAGE_MODEL))

    engine = ImageInferenceEngine(
        model_path=IMAGE_MODEL
    )

    print("\n✅ Image inference engine loaded successfully!")

    print("\nThe engine is ready to test:")
    print("CASE 1 → TEXT ONLY")
    print("CASE 2 → IMAGE ONLY")
    print("CASE 3 → TEXT + IMAGE")


if __name__ == "__main__":
    main()