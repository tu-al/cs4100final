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
from custom_svm import CustomLinearSVM
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns



DATASET_PATH = os.path.join("data", "fake_reviews.csv")
MODEL_OUTPUT_DIR = "models"

LOGREG_MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "logreg_pipeline.joblib")
ANN_MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "ann_pipeline.joblib")
SVM_MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "svm_pipeline.joblib")


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
            stop_words="english", # helps ignore basic filler words
            norm="l2"
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
            stop_words="english",
            norm="l2"
        )),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            max_iter=20,   # this can be modified
            verbose=True
        )),
    ])

def build_linear_svm_pipeline():
    return Pipeline([
        ("vectorizer", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            stop_words="english",
            norm="l2"
        )),
        ("classifier", CustomLinearSVM(
            lr=0.0005,
            C=1.0,
            max_iter=200
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

def save_diagnostic_plots(model_name, pipeline, X_test, y_test, output_dir="results/diagnostics"):
    os.makedirs(output_dir, exist_ok=True)

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["fake", "real"],
                yticklabels=["fake", "real"])
    plt.title(f"confusion matrix: {model_name}")
    plt.xlabel("predicted")
    plt.ylabel("true")
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    roc_scores = None
    if hasattr(pipeline, "predict_proba"):
        try:
            roc_scores = pipeline.predict_proba(X_test)[:, 1]
        except:
            pass
    elif hasattr(pipeline, "decision_function"):
        try:
            roc_scores = pipeline.decision_function(X_test)
        except:
            pass

    if roc_scores is not None:
        fpr, tpr, _ = roc_curve(y_test, roc_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("false positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC Curve- {model_name}")
        plt.legend()
        roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

    if roc_scores is not None:
        precision, recall, _ = precision_recall_curve(y_test, roc_scores)

        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title(f"Precision-Recall Curve- {model_name}")
        pr_path = os.path.join(output_dir, f"{model_name}_pr_curve.png")
        plt.tight_layout()
        plt.savefig(pr_path)
        plt.close()

    print(f"saved diagnostics for {model_name} in {output_dir}")


def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    raw_texts, raw_labels = load_dataset(DATASET_PATH)

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
    save_diagnostic_plots("Logistic Regression", logreg_pipeline, text_test, y_test)


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
    save_diagnostic_plots("ANN (MLPClassifier)", ann_pipeline, text_test, y_test)


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
    save_diagnostic_plots("Logistic Regression + Review Meta-Features", logreg_meta_pipeline, text_test, y_test)

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
    save_diagnostic_plots("Linear SVM", svm_pipeline, text_test, y_test)


    print("Saving Linear SVM model...")
    joblib.dump(
        {"pipeline": svm_pipeline, "label_encoder": label_encoder},
        SVM_MODEL_FILE
    )


    #save all metrics for run
    save_metrics_history(metrics_history)
    #save all plots for run
    save_plots_for_run(metrics_history)


    print("\nTraining complete. Models saved in:", MODEL_OUTPUT_DIR)




if __name__ == "__main__":
    main()
