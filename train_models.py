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
from sklearn.svm import LinearSVC
from datetime import datetime
import matplotlib.pyplot as plt
from length_weighted_logreg import build_length_weighted_logreg_pipeline



HOTEL_DATASET_PATH = os.path.join("data", "hotel_reviews.csv")
FAKE_DATASET_PATH  = os.path.join("data", "fake_reviews.csv")
MODEL_OUTPUT_DIR = "models"

LOGREG_MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "logreg_pipeline.joblib")
ANN_MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "ann_pipeline.joblib")
SVM_MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "svm_pipeline.joblib")
CUSTOM_LOGREG_MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "length_weighted_logreg_pipeline.joblib")



def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)

    if "text_" in df.columns:
        review_texts = df["text_"].astype(str)
    elif "review" in df.columns:
        review_texts = df["review"].astype(str)
    else:
        raise ValueError("Dataset missing text column")

    if "label" not in df.columns:
        raise ValueError("Dataset missing label column")

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
            hidden_layer_sizes=(256, 128),  # 2 layers: 256 -> 128
            activation="relu",
            solver="adam",
            max_iter=80,          
            alpha=1e-4, #note this line and the line below were limitations recommended by GPT as I encountered an overfitting problem
            learning_rate_init=1e-3, 
            batch_size=256,
            early_stopping=True,
            n_iter_no_change=5,
            verbose=True,
            random_state=42
        )),
    ])

def build_linear_svm_pipeline():
    #tf- idf + linear svm model
    return Pipeline([
        ("vectorizer", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1,2),
            stop_words="english",
        )),
        ("classifier", LinearSVC()),
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


#computes metrics for a model and returns a flat dict (so we can log to csv, it doesnt print anything)
def collect_metrics_for_model(model_name, trained_pipeline, X_test, y_test, class_names):
    y_pred = trained_pipeline.predict(X_test)

    #try to get some score for roc-auc
    scores = None
    if hasattr(trained_pipeline, "predict_proba"):
        try:
            proba = trained_pipeline.predict_proba(X_test)
            scores = proba[:,1] #positive class score
        except Exception:
            scores = None
    elif hasattr(trained_pipeline, "decision_function"):
        try:
            scores = trained_pipeline.decision_function(X_test)
        except Exception:
            scores = None
    
    #roc auc if possinle
    auc_score = None
    if scores is not None:
        try:
            auc_score = roc_auc_score(y_test, scores)
        except Exception:
            auc_score = None
    
    #dict classification report for log
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    metrics_entry = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "model": model_name,
        "accuracy": report_dict["accuracy"],
        "roc_auc": auc_score,
        f"precision_{class_names[0]}": report_dict[class_names[0]]['precision'],
        f"recall_{class_names[0]}": report_dict[class_names[0]]['recall'],
        f"f1_{class_names[0]}": report_dict[class_names[0]]['f1-score'],
        f"precision_{class_names[1]}": report_dict[class_names[1]]['precision'],
        f"recall_{class_names[1]}": report_dict[class_names[1]]['recall'],
        f"f1_{class_names[1]}": report_dict[class_names[1]]['f1-score'],
    }
    return metrics_entry

#appending a list of metrics to a csv file 
def save_metrics_history(metrics_list, output_dir="results", filename="metrics_history.csv"):
    if not metrics_list:
        return
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)

    df = pd.DataFrame(metrics_list)

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    
    print(f"\nSaved metrics for {len(metrics_list)} models to {csv_path}")


def save_plots_for_run(metrics_list, output_dir="results/plots"):
    #create and save accuracy and ROC-AUC plots for this run only
    #saves two png files
    if not metrics_list:
        return

    os.makedirs(output_dir, exist_ok=True)

    #convert to df
    df = pd.DataFrame(metrics_list)

    #unique timestamp for each run
    ts = df.iloc[0]["timestamp"].replace(":", "-")

    #accuracy plot
    plt.figure()
    plt.bar(df["model"], df["accuracy"])
    plt.title(f"Accuracy per Model (Run {ts})")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    acc_path = os.path.join(output_dir, f"accuracy_run_{ts}.png")
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    # roc auc plot
    if "roc_auc" in df.columns:
        plt.figure()
        plt.bar(df["model"], df["roc_auc"])
        plt.title(f"ROC-AUC per Model (Run {ts})")
        plt.ylabel("ROC-AUC")
        plt.xticks(rotation=45, ha="right")
        auc_path = os.path.join(output_dir, f"roc_auc_run_{ts}.png")
        plt.tight_layout()
        plt.savefig(auc_path)
        plt.close()

    print(f"Saved run plots to {output_dir}")


def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print("Loading datasets...")

    text1, labels1 = load_dataset("data/fake_reviews.csv")
    text2, labels2 = load_dataset("data/hotel_reviews.csv")

    # Merge datasets
    raw_texts = pd.concat([text1, text2], ignore_index=True)
    raw_labels = pd.concat([labels1, labels2], ignore_index=True)

    print("Dataset sizes:")
    print("  fake_reviews.csv:", len(text1))
    print("  hotel_reviews.csv:", len(text2))
    print("  TOTAL:", len(raw_texts))

    print("Encoding labels...")
    numeric_labels, label_encoder = encode_labels_as_numbers(raw_labels)

    class_names = list(label_encoder.classes_)
    metrics_history = []

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

    metrics_history.append(
        collect_metrics_for_model("Logistic Regression", logreg_pipeline, text_test, y_test, class_names)
    )

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

    metrics_history.append(
        collect_metrics_for_model("ANN (MLPClassifier)", ann_pipeline, text_test, y_test, class_names)
    )

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

    metrics_history.append(
        collect_metrics_for_model("Logistic Regression + Review Meta-Features", logreg_meta_pipeline, text_test, y_test, class_names)
    )

    meta_model_path = os.path.join(MODEL_OUTPUT_DIR, "logreg_meta_pipeline.joblib")
    print("Saving LogReg + Meta-Features model to:", meta_model_path)

    joblib.dump(
        {"pipeline": logreg_meta_pipeline, "label_encoder": label_encoder},
        meta_model_path
    )

    # train linear svm
    print("\nTraining Linear SVM model...")
    svm_pipeline = build_linear_svm_pipeline()
    svm_pipeline.fit(text_train, y_train)
    evaluate_model("Linear SVM", svm_pipeline, text_test, y_test)

    metrics_history.append(
        collect_metrics_for_model("Linear SVM", svm_pipeline, text_test, y_test, class_names)
    )

    print("Saving Linear SVM model...")
    joblib.dump(
        {"pipeline": svm_pipeline, "label_encoder": label_encoder},
        SVM_MODEL_FILE
    )

    # ---- Train Custom Length-Weighted Logistic Regression ----
    print("\nTraining Length-Weighted Logistic Regression (Custom)...")

    # Compute lengths for BOTH train + test
    train_lengths = text_train.apply(lambda t: len(t.split()))
    test_lengths  = text_test.apply(lambda t: len(t.split()))

    custom_pipeline = build_length_weighted_logreg_pipeline()

    # IMPORTANT: pass lengths manually
    custom_pipeline.named_steps["classifier"].fit(
        custom_pipeline.named_steps["vectorizer"].fit_transform(text_train),
        y_train,
        lengths=train_lengths
    )

    print("Evaluating Custom Logistic Regression...")
    evaluate_model("Custom Length-Weighted LogReg", custom_pipeline, text_test, y_test)

    metrics_history.append(
        collect_metrics_for_model(
            "Custom Length-Weighted LogReg",
            custom_pipeline,
            text_test,
            y_test,
            class_names
        )
    )

    print("Saving Custom Logistic Regression model...")
    joblib.dump(
        {"pipeline": custom_pipeline, "label_encoder": label_encoder},
        CUSTOM_LOGREG_MODEL_FILE
    )



    #save all metrics for run
    save_metrics_history(metrics_history)
    #save all plots for run
    save_plots_for_run(metrics_history)


    print("\nTraining complete. Models saved in:", MODEL_OUTPUT_DIR)




if __name__ == "__main__":
    main()
