import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import analyze_image, get_stats_from_history

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed_extensions = {"jpg", "jpeg", "png", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Use JPG, PNG, or WEBP."}), 400

    try:
        image_bytes = file.read()
        result, confidence = analyze_image(image_bytes)

        entry = {
            "id": datetime.utcnow().strftime("%Y%m%d%H%M%S%f"),
            "result": result,
            "confidence": round(confidence, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "filename": file.filename,
        }

        history = load_history()
        history.insert(0, entry)
        history = history[:200]
        save_history(history)

        return jsonify({
            "result": result,
            "confidence": round(confidence, 2),
            "timestamp": entry["timestamp"],
            "filename": file.filename,
        })
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def get_history():
    history = load_history()
    return jsonify(history)


@app.route("/stats", methods=["GET"])
def get_stats():
    history = load_history()
    stats = get_stats_from_history(history)
    return jsonify(stats)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "DeepShield AI"})


@app.route("/history/clear", methods=["DELETE"])
def clear_history():
    save_history([])
    return jsonify({"message": "History cleared"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
