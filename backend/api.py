import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from model_utils import ModelManager, ImageProcessor

# ------------------------------
# Global objects (loaded once)
# ------------------------------
app = Flask(__name__)
CORS(app)

manager = ModelManager()
processor = ImageProcessor()   # ✅ define once globally, not inside route

manager = ModelManager()
processor = ImageProcessor()

# ✅ Load a model once at startup
try:
    manager.get_best_model()   # will set manager.predictor
except Exception as e:
    print("Error loading model:", e)



# ------------------------------
# Helper: decode base64 image
# ------------------------------
def decode_image(img_base64: str):
    nparr = np.frombuffer(base64.b64decode(img_base64), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ------------------------------
# API Routes
# ------------------------------
@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        img_data = data["image"].split(",")[1]
        frame = decode_image(img_data)

        # ✅ Use global processor to detect + crop faces
        faces = processor.process_image_for_emotion(frame)

        if not faces:
            return jsonify({"results": []})  # no faces found

        results = []
        for face_img, coords in faces:
            prediction = manager.predictor.predict_emotion(
                face_img, return_probabilities=True
            )
            results.append({
                "bbox": coords,
                "emotion": prediction["emotion"],
                "confidence": prediction["confidence"],
                "probabilities": prediction.get("probabilities", {})
            })

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download_debug_face", methods=["GET"])
def download_debug_face():
    """Optional: allow downloading the last debug face crop"""
    debug_path = os.path.join(os.path.dirname(__file__), "debug_face.jpg")
    if os.path.exists(debug_path):
        return send_file(debug_path, mimetype="image/jpeg", as_attachment=True)
    return jsonify({"error": "No debug face available"}), 404


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
