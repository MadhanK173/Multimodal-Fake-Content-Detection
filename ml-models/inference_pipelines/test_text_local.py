from text_inference import TextInferenceEngine


def main():
    print("=" * 70)
    print("LOCAL TEXT MODEL TEST")
    print("=" * 70)

    try:
        print("\nInitializing RoBERTa...")
        engine = TextInferenceEngine()

        text_sample = """
        Artificial intelligence is transforming industries by automating
        tasks, improving decision-making, and enabling new technological innovations.
        """

        print("\nRunning prediction...")

        result = engine.predict(text_sample)

        print("\nPREDICTION RESULT")
        print("-" * 70)

        for key, value in result.items():
            print(f"{key}: {value}")

        print("\n" + "=" * 70)
        print("TEXT MODEL TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as exc:
        print("\n" + "=" * 70)
        print("TEXT MODEL TEST FAILED")
        print("=" * 70)
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    main()