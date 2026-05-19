import os
import logging
from flask import Flask, render_template, request, jsonify
import joblib

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Model loading ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model      = joblib.load(os.path.join(BASE_DIR, "sms_svm_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
    logger.info("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    logger.critical("Could not load model files: %s", e)
    raise SystemExit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_SMS_LENGTH = 1600   # ~10 concatenated SMS segments


# ── Page routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", active_page="home")

@app.route("/model")
def model_page():
    return render_template("model.html", active_page="model")

@app.route("/dataset")
def dataset_page():
    return render_template("dataset.html", active_page="dataset")

@app.route("/about")
def about_page():
    return render_template("about.html", active_page="about")


# ── Prediction route ──────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    # 1. Parse JSON body
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    sms = payload.get("sms", "").strip()

    # 2. Input validation
    if not sms:
        return jsonify({"error": "The 'sms' field is required and cannot be empty."}), 422

    if len(sms) > MAX_SMS_LENGTH:
        return jsonify({
            "error": f"Message too long. Maximum allowed length is {MAX_SMS_LENGTH} characters."
        }), 422

    # 3. Classify
    try:
        sms_vec = vectorizer.transform([sms])
        pred    = model.predict(sms_vec)[0]
    except Exception as e:
        logger.exception("Prediction failed for input: %.80r", sms)
        return jsonify({"error": "Prediction failed. Please try again."}), 500

    # 4. Build response
    if pred == 1:
        result, label = "Spam / Smishing", "threat"
    else:
        result, label = "Legitimate SMS", "safe"

    logger.info("Predicted '%s' for message (len=%d)", label, len(sms))

    return jsonify({"prediction": result, "label": label})


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    # 404 gets no active_page — no nav link should be highlighted
    return render_template("404.html"), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405

@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled server error")
    return jsonify({"error": "An unexpected server error occurred."}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=5000)