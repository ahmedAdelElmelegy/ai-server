"""
🎯 Cardiac Surgery Complication Prediction API - Production Optimized
• Ensemble (Random Forest + XGBoost)
• Pretrained model loading only
• Secure, fast, and scalable API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Globals
model = None
label_encoders = {}
feature_columns = []
target_encoder = None
scaler = None


# =========================
# Model Loader
# =========================
def load_model():
    global model, label_encoders, feature_columns, target_encoder, scaler

    required_files = [
        "rf_xgb_model.pkl",
        "label_encoders.pkl",
        "feature_columns.pkl",
        "target_encoder.pkl",
        "scaler.pkl",
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"❌ Missing required file: {file}")

    model = joblib.load("rf_xgb_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    print("✅ Model and encoders loaded successfully")


# =========================
# Health Check
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


# =========================
# Feature Metadata
# =========================
@app.route("/features", methods=["GET"])
def features():
    try:
        return jsonify({
            "features": feature_columns,
            "options": {k: v.classes_.tolist() for k, v in label_encoders.items()},
            "complications": target_encoder.classes_.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Prediction Endpoint
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        missing = [f for f in feature_columns if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        df = pd.DataFrame([data])[feature_columns]

        # Encode categorical features
        for col, encoder in label_encoders.items():
            if col in df.columns:
                if df[col].iloc[0] not in encoder.classes_:
                    df[col] = encoder.classes_[0]
                df[col] = encoder.transform(df[col])

        # Scale numeric features
        if "Age" in df.columns:
            df["Age"] = scaler.transform(df[["Age"]])

        # Predict
        probs = model.predict_proba(df)[0]
        pred_idx = probs.argmax()
        prediction = target_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "confidence_level": (
                "Very High" if confidence >= 0.8 else
                "High" if confidence >= 0.6 else
                "Moderate" if confidence >= 0.4 else
                "Low"
            ),
            "all_probabilities": {
                target_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(probs)
            },
            "is_high_confidence": confidence >= 0.7
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Model Info
# =========================
@app.route("/model-info", methods=["GET"])
def model_info():
    try:
        rf = model.estimators_[0]
        return jsonify({
            "model_type": "Ensemble (Random Forest + XGBoost)",
            "features": feature_columns,
            "classes": target_encoder.classes_.tolist(),
            "feature_importance": {
                f: float(i) for f, i in zip(feature_columns, rf.feature_importances_)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# App Start
# =========================
if __name__ == "__main__":
    print("🏥 Cardiac Surgery Complication Prediction API")
    load_model()
    port = int(os.environ.get("PORT", 5000))  # يستخدم بورت Koyeb لو متاح
    app.run(host="0.0.0.0", port=port)

