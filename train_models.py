import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from review_features import ReviewFeatureExtractor
import joblib

DATASET_PATH = os.path.join("data", "fake_reviews.csv")
MODEL_OUTPUT_DIR = "models"

LOGREG_MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "logreg_pipeline.joblib")
ANN_MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "ann_pipeline.joblib")


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)

    # Extract review text
    review_texts = df["text_"].astype(str)

    # Convert raw labels ('CG', 'OR') → ('fake', 'real') note to change this or comment out if we are not using these.
    human_label_map = {
        "CG": "fake",
        "OR": "real",
    }
    review_labels = df["label"].map(human_label_map).astype(str)

    return review_texts, review_labels


def encode_labels_as_numbers(label_strings):
    encoder = LabelEncoder()
    numeric_labels = encoder.fit_transform(label_strings)

    print("Label mapping:")
    for index, class_name in enumerate(encoder.classes_):
        print(f"  {index} -> {class_name}")

    return numeric_labels, encoder


#model maker

#it will be noted that this small function here was made with the help of AI, I needed a basis on what a pipline is 
#supposed to look like so I asked it to give me an example of what one would look like. 
def build_logistic_regression_pipeline():
    return Pipeline([
        ("vectorizer", TfidfVectorizer( # this mainly helps converting text to numbers
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english" # helps ignore basic filler words
        )),
        ("classifier", LogisticRegression(
            max_iter=300,
            n_jobs=-1
        )),
    ])


#note this one is based off the function on the top, but I did type this completely manually.
#as it may count as AI use since the first one needed AI help hence this note. 
def build_ann_pipeline():
    return Pipeline([
        ("vectorizer", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            max_iter=20,   # this can be modified
            verbose=True
        )),
    ])

def build_logreg_with_review_features():
    
    text_branch = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english",
        )),
    ])

    meta_branch = Pipeline([
        ("review_features", ReviewFeatureExtractor()),
    ])

    combined = FeatureUnion([
        ("text", text_branch),
        ("meta", meta_branch),
    ])

    classifier = LogisticRegression(
        max_iter=300,
        n_jobs=-1,
    )

    return Pipeline([
        ("features", combined),
        ("classifier", classifier),
    ])


def evaluate_model(model_name, trained_pipeline, X_test, y_test):
 
    print(f"\n====== Evaluating {model_name} ======")

    predicted_labels = trained_pipeline.predict(X_test)

    # Probability-based scoring
    if hasattr(trained_pipeline, "predict_proba"):
        try:
            predicted_probabilities = trained_pipeline.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, predicted_probabilities)
            print(f"ROC-AUC: {auc_score:.4f}")
        except Exception:
            print("Could not compute ROC-AUC.")

    print(classification_report(y_test, predicted_labels))

def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    raw_texts, raw_labels = load_dataset(DATASET_PATH)

    print("Encoding labels...")
    numeric_labels, label_encoder = encode_labels_as_numbers(raw_labels)

    print("Splitting data into train/test...")
    text_train, text_test, y_train, y_test = train_test_split(
        raw_texts,
        numeric_labels,
        test_size=0.2,
        random_state=42,
        stratify=numeric_labels
    )

    # ---- Train Logistic Regression ----
    print("\nTraining Logistic Regression model...")
    logreg_pipeline = build_logistic_regression_pipeline()
    logreg_pipeline.fit(text_train, y_train)
    evaluate_model("Logistic Regression", logreg_pipeline, text_test, y_test)

    print("Saving Logistic Regression model...")
    joblib.dump(
        {"pipeline": logreg_pipeline, "label_encoder": label_encoder},
        LOGREG_MODEL_FILE
    )

    # ---- Train ANN ----
    print("\nTraining ANN model...")
    ann_pipeline = build_ann_pipeline()
    ann_pipeline.fit(text_train, y_train)
    evaluate_model("ANN (MLPClassifier)", ann_pipeline, text_test, y_test)

    print("Saving ANN model...")
    joblib.dump(
        {"pipeline": ann_pipeline, "label_encoder": label_encoder},
        ANN_MODEL_FILE
    )

    # ---- Train Logistic Regression + Review Meta-Features ----
    print("\nTraining Logistic Regression + Review Meta-Features...")
    logreg_meta_pipeline = build_logreg_with_review_features()
    logreg_meta_pipeline.fit(text_train, y_train)

    evaluate_model(
        "Logistic Regression + Review Meta-Features",
        logreg_meta_pipeline,
        text_test,
        y_test
    )

    meta_model_path = os.path.join(MODEL_OUTPUT_DIR, "logreg_meta_pipeline.joblib")
    print("Saving LogReg + Meta-Features model to:", meta_model_path)

    joblib.dump(
        {"pipeline": logreg_meta_pipeline, "label_encoder": label_encoder},
        meta_model_path
    )

    print("\nTraining complete. Models saved in:", MODEL_OUTPUT_DIR)




if __name__ == "__main__":
    main()
