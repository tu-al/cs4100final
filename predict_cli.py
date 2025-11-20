import os
import joblib

# Path to the saved logistic regression pipeline
MODEL_FILE_PATH = os.path.join("models", "logreg_meta_pipeline.joblib")


def load_trained_pipeline(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Run train_models.py first to create it."
        )

    saved_bundle = joblib.load(model_path)
    trained_pipeline = saved_bundle["pipeline"]
    label_encoder = saved_bundle["label_encoder"]
    return trained_pipeline, label_encoder


def predict_single_review(review_text: str):
    trained_pipeline, label_encoder = load_trained_pipeline(MODEL_FILE_PATH)

    # Get probability for each class (e.g. [P(class0), P(class1)])
    class_probabilities = trained_pipeline.predict_proba([review_text])[0]

    class_names = list(label_encoder.classes_)  # e.g. ["fake", "real"] or ["real", "fake"]

    # Find which index corresponds to "fake"
    if "fake" in class_names:
        fake_index = class_names.index("fake")
    else:
        # Fallback if encoder classes are ordered differently than expected
        fake_index = 1

    prob_fake = float(class_probabilities[fake_index])
    prob_real = 1.0 - prob_fake

    # Predicted label is the class with the highest probability
    best_index = int(class_probabilities.argmax())
    predicted_label = class_names[best_index]

    return prob_real, prob_fake, predicted_label


def interactive_cli():
    print("Fake Review Detector (Logistic Regression)")
    print("Type or paste a review, then press Enter.")
    print("Press Enter on an empty line to quit.")

    while True:
        try:
            user_input = input("\nReview: ").strip()
        except EOFError:
            # Handles Ctrl+D / unexpected input stream end
            print("\nEOF received. Exiting.")
            break

        if user_input == "":
            print("Goodbye.")
            break

        prob_real, prob_fake, label = predict_single_review(user_input)

        print(f"\nPredicted label: {label.upper()}")
        print(f"Estimated probability REAL: {prob_real:.2%}")
        print(f"Estimated probability FAKE: {prob_fake:.2%}")


if __name__ == "__main__":
    interactive_cli()
