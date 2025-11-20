import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Adjust these if your filenames are slightly different
DATASET_PATH = os.path.join("data", "fake_reviews.csv")
MODEL_PATH = os.path.join("models", "logreg_pipeline.joblib")


def load_dataset():
    #this right here should be the translater for the labels
    df = pd.read_csv(DATASET_PATH)

    texts = df["text_"].astype(str)

    label_map = {
        "CG": "fake",
        "OR": "real",
    }
    labels_str = df["label"].map(label_map).astype(str)

    return texts, labels_str


def main():
    # ----- Load data -----
    print("Loading dataset...")
    texts, labels_str = load_dataset()

    # ----- Load trained model + encoder -----
    print("Loading trained model...")
    saved = joblib.load(MODEL_PATH)
    pipeline = saved["pipeline"]
    label_encoder = saved["label_encoder"]

    # Encode labels using the SAME encoder mapping as training
    labels_num = label_encoder.transform(labels_str)

    # ----- Train/test split (must match train_models.py) -----
    text_train, text_test, y_train, y_test = train_test_split(
        texts,
        labels_num,
        test_size=0.2,
        random_state=42,
        stratify=labels_num,
    )

    # ----- Run model on test set -----
    print("Running predictions on test split...")
    proba = pipeline.predict_proba(text_test)
    y_pred = proba.argmax(axis=1)

    classes = label_encoder.classes_  # e.g. ['fake', 'real']
    print("Classes in encoder:", classes)

    # Figure out which index is 'fake' and which is 'real'
    fake_idx = list(classes).index("fake")
    real_idx = list(classes).index("real")

    # Confidence of each prediction (max probability)
    pred_confidence = proba.max(axis=1)

    # ----- Masks for FP / FN -----
    # false positive = predicted fake, actually real
    fp_mask = (y_test == real_idx) & (y_pred == fake_idx)

    # false negative = predicted real, actually fake
    fn_mask = (y_test == fake_idx) & (y_pred == real_idx)

    print(f"\nTotal test examples: {len(y_test)}")
    print(f"False positives: {fp_mask.sum()}")
    print(f"False negatives: {fn_mask.sum()}")

    # ----- Helper to print examples -----
    def show_examples(mask, title, max_n=20):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            print(f"\n===== {title} =====")
            print("None found.")
            return

        # Sort by confidence descending (most confident mistakes first)
        idxs_sorted = idxs[np.argsort(pred_confidence[idxs])[::-1]]

        print(f"\n===== {title} (showing up to {max_n}, total {len(idxs)}) =====")
        for rank, i in enumerate(idxs_sorted[:max_n], start=1):
            true_label = classes[y_test[i]]
            pred_label = classes[y_pred[i]]
            p_fake = proba[i, fake_idx]
            p_real = proba[i, real_idx]

            print(f"\n#{rank}")
            print(f"True label: {true_label} | Predicted: {pred_label}")
            print(f"P(real) = {p_real:.3f}, P(fake) = {p_fake:.3f}")
            print("Text:")
            print(text_test.iloc[i])

    show_examples(fp_mask, "FALSE POSITIVES (predicted FAKE, actually REAL)")
    show_examples(fn_mask, "FALSE NEGATIVES (predicted REAL, actually FAKE)")


if __name__ == "__main__":
    main()
