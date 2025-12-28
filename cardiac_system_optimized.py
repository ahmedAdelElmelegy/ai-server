"""
========================================
CARDIAC SURGERY COMPLICATION PREDICTION SYSTEM
Optimized with 3-Layer Decision Logic
========================================
"""

import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ================================
# Predictor Class
# ================================
class CardiacComplicationPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None

        # Medical priority (1 = highest risk)
        self.medical_priority = {
            'Myocardial_Infarction': 1,
            'Cardiac_Arrhythmias': 2,
            'Heart_Failure': 3,
            'Thromboembolism': 4,
            'Hemorrhage_Bleeding': 5,
            'Acute_Kidney_Injury': 6,
            'Atelectasis_Pneumonia': 7,
            'Delirium': 8,
            'Wound_Infection': 9
        }

        # Complication signs
        self.complications = {
            'Delirium': ['Disorientation', 'Agitation', 'Altered_level_of_consciousness'],
            'Acute_Kidney_Injury': ['Decreased_urine_output_Aki', 'Fluid_retention', 'Elevated_creatinine', 'High_BP', 'Nausea_and_vomiting'],
            'Atelectasis_Pneumonia': ['Dyspnea', 'Decreased_oxygen_saturation', 'Crackles'],
            'Hemorrhage_Bleeding': ['Excessive_surgical_site_bleeding', 'Hypotension', 'Tachycardia', 'Decreased_hemoglobin_level', 'Signs_of_shock'],
            'Myocardial_Infarction': ['Chest_pain_radiating_to_arm_or_jaw', 'Diaphoresis_excessive_sweating', 'Shortness_of_breath', 'Anxiety'],
            'Thromboembolism': ['Chest_pain', 'Sudden_shortness_of_breath', 'Cyanosis', 'Cold_extremities_Limb_pain', 'Weak_or_absent_pulse'],
            'Cardiac_Arrhythmias': ['Palpitations', 'Dizziness', 'Syncope_fainting', 'Fatigue', 'Shortness_of_breath_Arrhythmia'],
            'Heart_Failure': ['Shortness_of_breath_dyspnea', 'Peripheral_edema_swelling_of_feet_legs', 'Severe_fatigue', 'Rapid_weight_gain', 'Decreased_urine_output_hf'],
            'Wound_Infection': ['Fever_Wound', 'Redness_around_the_wound', 'Purulent_pus_discharge', 'Localized_pain', 'Swelling']
        }

    # -----------------------------
    # 3-Layer Decision Logic
    # -----------------------------
    def predict_with_logic(self, signs_dict):
        active_signs = [s for s, active in signs_dict.items() if active]
        if not active_signs:
            return self._empty_result()

        # Layer 1: Exact match
        possible = set()
        for sign in active_signs:
            for comp, signs_list in self.complications.items():
                if sign in signs_list:
                    possible.add(comp)
        if len(possible) == 1:
            comp = list(possible)[0]
            return self._format_result(comp, active_signs, layer="Layer 1: Exact Match")

        # Layer 2: Scoring
        scores = {}
        for comp, signs_list in self.complications.items():
            matching = [s for s in active_signs if s in signs_list]
            scores[comp] = {
                'score': len(matching),
                'confidence': len(matching)/len(signs_list) if signs_list else 0,
                'matching_signs': matching,
                'total_signs': len(signs_list)
            }
        max_score = max([v['score'] for v in scores.values()])
        if max_score == 0:
            return self._unknown_result()
        top_complications = [c for c, v in scores.items() if v['score'] == max_score]
        if len(top_complications) == 1:
            comp = top_complications[0]
            return self._format_result(comp, active_signs, scores=scores, layer="Layer 2: Highest Score")

        # Layer 3: Priority
        best_comp = min(top_complications, key=lambda c: self.medical_priority.get(c, 99))
        return self._format_result(best_comp, active_signs, scores=scores,
                                   layer=f"Layer 3: Medical Priority (tie {len(top_complications)})")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _get_confidence_level(self, confidence):
        if confidence >= 0.80: return "Very High"
        if confidence >= 0.65: return "High"
        if confidence >= 0.50: return "Moderate"
        return "Low"

    def _format_result(self, comp, active_signs, scores=None, layer=""):
        if scores is None:
            matching = len([s for s in active_signs if s in self.complications[comp]])
            total = len(self.complications[comp])
            confidence = matching / total if total > 0 else 0
        else:
            confidence = scores[comp]['confidence']
            matching = scores[comp]['score']
            total = scores[comp]['total_signs']
        return {
            "prediction": comp,
            "confidence": confidence,
            "confidence_level": self._get_confidence_level(confidence),
            "decision_layer": layer,
            "score": matching,
            "matching_signs": matching,
            "total_signs": total
        }

    def _empty_result(self):
        return {
            "prediction": "No_Complication",
            "confidence": 1.0,
            "confidence_level": "Very High",
            "decision_layer": "No Signs Detected",
            "score": 0,
            "matching_signs": 0,
            "total_signs": 0
        }

    def _unknown_result(self):
        return {
            "prediction": "Unknown_Pattern",
            "confidence": 0.0,
            "confidence_level": "Low",
            "decision_layer": "No Match Found",
            "score": 0,
            "matching_signs": 0,
            "total_signs": 0
        }

    # -----------------------------
    # Data generation & training
    # -----------------------------
    def generate_synthetic_data(self, n_samples=5000):
        all_signs = sorted(list(set(sign for signs in self.complications.values() for sign in signs)))
        self.feature_names = all_signs
        data, labels = [], []
        for comp, primary in self.complications.items():
            per_comp = n_samples // len(self.complications)
            for _ in range(per_comp):
                sample = {s: False for s in all_signs}
                for s in primary: sample[s] = np.random.random() < 0.85
                for s in all_signs:
                    if s not in primary: sample[s] = np.random.random() < 0.15
                data.append(list(sample.values()))
                labels.append(comp)
        return np.array(data), np.array(labels)

    def train(self):
        X, y = self.generate_synthetic_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {(y_pred==y_test).mean():.2%}")
        return self.model

    def save_model(self, filepath='cardiac_model_optimized.pkl'):
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'complications': self.complications,
            'medical_priority': self.medical_priority
        }, filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath='cardiac_model_optimized.pkl'):
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.complications = data['complications']
        self.medical_priority = data.get('medical_priority', self.medical_priority)
        print(f"✓ Model loaded from {filepath}")


# ================================
# Flask App
# ================================
def create_app():
    app = Flask(__name__)
    CORS(app)

    predictor = CardiacComplicationPredictor()
    if os.path.exists('cardiac_model_optimized.pkl'):
        predictor.load_model('cardiac_model_optimized.pkl')
    else:
        print("⚠️ Model not found! Run: python cardiac_system_optimized.py train")

    # --- Routes ---
    @app.route('/')
    def home(): return "Cardiac Surgery Complication Prediction API is running"
    @app.route('/favicon.ico')
    def favicon(): return "", 204

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'model_loaded': predictor.model is not None,
            'total_features': len(predictor.feature_names),
            'complications': list(predictor.complications.keys()),
            'decision_layers': 3
        })

    @app.route('/api/features', methods=['GET'])
    def get_features():
        return jsonify({'features': predictor.feature_names, 'count': len(predictor.feature_names)})

    @app.route('/api/complications', methods=['GET'])
    def get_complications():
        return jsonify({'complications': predictor.complications, 'medical_priority': predictor.medical_priority})

    @app.route('/api/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            if not data or 'signs' not in data:
                return jsonify({'error': 'Missing "signs" field in request body'}), 400
            signs_dict = data['signs']
            if not isinstance(signs_dict, dict):
                return jsonify({'error': '"signs" must be a dictionary'}), 400
            result = predictor.predict_with_logic(signs_dict)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


# ================================
# CLI Entrypoint
# ================================
def train_model():
    print("="*60)
    print("CARDIAC COMPLICATION PREDICTION MODEL - TRAINING")
    print("="*60)
    predictor = CardiacComplicationPredictor()
    predictor.train()
    predictor.save_model()
    print("\nTraining complete! Model ready for API.\n")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':

        train_model()
    else:
        app = create_app()
        port = int(os.environ.get("PORT", 5000))
        print(f"Starting server on port {port}...")
        app.run(host="0.0.0.0", port=port)
