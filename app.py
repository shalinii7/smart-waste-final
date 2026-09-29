from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import json, uuid, datetime

app = Flask(__name__)
CORS(app)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "waste_model1.keras"
CLASSES_PATH = BASE / "model" / "class_names.json"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
class_names = json.loads(CLASSES_PATH.read_text())
print(f"Model loaded: {class_names}")

BIN_INFO = {
    "cardboard": {"bin": "Blue Bin", "type": "Recyclable", "tip": "Flatten before disposal"},
    "glass": {"bin": "Green Bin", "type": "Glass Waste", "tip": "Rinse bottle"},
    "metal": {"bin": "Grey Bin", "type": "Metal Waste", "tip": "Empty contents"},
    "organic": {"bin": "Green Bin", "type": "Compost", "tip": "Compostable"},
    "paper": {"bin": "Blue Bin", "type": "Paper Waste", "tip": "Keep dry"},
    "plastic": {"bin": "Red Bin", "type": "Plastic Waste", "tip": "Rinse and recycle"}
}

UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

def predict_image(img: Image.Image):
    img = img.convert('RGB').resize((224,224))
    arr = preprocess_input(np.array(img))
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return {
        "predicted": class_names[idx],
        "confidence": round(float(preds[idx]*100),2),
        "all_scores": {class_names[i]: round(float(p*100),2) for i,p in enumerate(preds)}
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "waste_model1.keras", "accuracy": "89.24%", "classes": class_names})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        img = Image.open(file.stream)
        result = predict_image(img)

        label = result["predicted"]
        conf = result["confidence"]

        # NEW LOGIC - Fixes your red box bug
        if conf < 75:
            bin_name = "⚠️ Manual Check Needed"
            bin_type = f"Low confidence - looks like {label} ({conf}%)"
            tip_text = "Try with white background or better lighting"
            color = "#FF9800"
            final_info = {"bin": bin_name, "type": bin_type, "tip": tip_text}
        else:
            final_info = BIN_INFO.get(label, {})
            bin_name = final_info.get("bin", "")
            bin_type = final_info.get("type", "")
            tip_text = final_info.get("tip", "")
            if label == "plastic":
                color = "#F44336"
            elif label in ["paper", "cardboard"]:
                color = "#2196F3"
            else:
                color = "#4CAF50"

        log_entry = {
            "id": str(uuid.uuid4())[:8],
            "time": datetime.datetime.now().isoformat(),
            "file": file.filename,
            **result,
            "bin_info": final_info
        }
        with open(UPLOAD_DIR / "history.jsonl", "a") as f:
            f.write(json.dumps(log_entry)+"\n")

        return jsonify({
            "predicted": label,
            "confidence": conf,
            "bin": bin_name,
            "type": bin_type,
            "tip": tip_text,
            "color": color,
            "all": result["all_scores"],
            "id": log_entry["id"]
        })
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)