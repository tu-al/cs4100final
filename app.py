# app.py
import os
import joblib
from flask import Flask, request, jsonify, render_template

MODEL_DIR = "models"

LOGREG_BASIC_PATH = os.path.join(MODEL_DIR, "logreg_pipeline.joblib")
ANN_PATH = os.path.join(MODEL_DIR, "ann_pipeline.joblib")
SVM_PATH = os.path.join(MODEL_DIR, "svm_pipeline.joblib")
LOGREG_META_PATH = os.path.join(MODEL_DIR, "logreg_meta_pipeline.joblib")
CUSTOM_LOGREG_PATH = os.path.join(
    MODEL_DIR,
    "length_weighted_logreg_pipeline.joblib"
)

app = Flask(__name__)

MODEL_BUNDLES = {}


def load_bundle(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    saved = joblib.load(model_path)
    pipeline = saved["pipeline"]
    label_encoder = saved["label_encoder"]
    return pipeline, label_encoder


def load_all_models():
    """Load the trained models into memory once at startup."""
    print("Loading models...")
    MODEL_BUNDLES["logreg_meta"] = load_bundle(LOGREG_META_PATH)
    MODEL_BUNDLES["logreg_basic"] = load_bundle(LOGREG_BASIC_PATH)
    MODEL_BUNDLES["ann"] = load_bundle(ANN_PATH)
    MODEL_BUNDLES["svm"] = load_bundle(SVM_PATH)

    MODEL_BUNDLES["logreg_custom"] = load_bundle(CUSTOM_LOGREG_PATH)

    print("Loaded models:", ", ".join(MODEL_BUNDLES.keys()))


def predict_with_bundle(text: str, pipeline, label_encoder):
    """Return (prob_real, prob_fake, predicted_label) for a single review."""
    y_pred_numeric = pipeline.predict([text])[0]
    predicted_label = label_encoder.inverse_transform([int(y_pred_numeric)])[0]
    class_names = list(label_encoder.classes_)

    prob_real = None
    prob_fake = None

    if hasattr(pipeline, "predict_proba"):
        class_probas = pipeline.predict_proba([text])[0]

        if "fake" in class_names:
            fake_index = class_names.index("fake")
        else:
            fake_index = 1

        prob_fake = float(class_probas[fake_index])
        prob_real = 1.0 - prob_fake

    return prob_real, prob_fake, predicted_label


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Request JSON:
      {
        "text": "...",
        "model": "logreg_meta" | "ann" | "svm" |
                  "logreg_basic" | "logreg_custom"
      }
    """
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    model_name = data.get("model", "logreg_meta")

    if text == "":
        return jsonify({"error": "Missing or empty 'text' field."}), 400

    if model_name not in MODEL_BUNDLES:
        return jsonify({"error": f"Unknown model '{model_name}'."}), 400

    pipeline, label_encoder = MODEL_BUNDLES[model_name]
    prob_real, prob_fake, label = predict_with_bundle(text, pipeline, label_encoder)

    confidence = None
    if prob_real is not None and prob_fake is not None:
        label_lower = str(label).lower()
        if label_lower == "real":
            confidence = prob_real
        elif label_lower == "fake":
            confidence = prob_fake
        else:
            confidence = max(prob_real, prob_fake)

    return jsonify({
        "model": model_name,
        "label": label,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "confidence": confidence,
        "text_length": len(text),
    })


#this part here is for locally run instances.
if __name__ == "__main__":
    load_all_models()
    app.run(host="0.0.0.0", port=5001, debug=True)
