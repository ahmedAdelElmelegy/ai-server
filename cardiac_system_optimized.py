"""
========================================
CARDIAC SURGERY COMPLICATION PREDICTION API
Full Parameters - Matches Dart Model
PythonAnywhere (WSGI Ready)
========================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

# ======================================
# Predictor
# ======================================
class CardiacComplicationPredictor:
    def __init__(self):
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

        self.complications = {
            'Delirium': [
                'Disorientation', 'Agitation',
                'Altered_level_of_consciousness',
                'Restlessness', 'Visual'
            ],
            'Acute_Kidney_Injury': [
                'Decreased_urine_output_Aki', 'Fluid_retention',
                'Elevated_creatinine', 'High_BP', 'Nausea_and_vomiting'
            ],
            'Atelectasis_Pneumonia': [
                'Dyspnea', 'Decreased_oxygen_saturation',
                'Crackles', 'Cough_atelectasis',
                'Fever_atelectasis'
            ],
            'Hemorrhage_Bleeding': [
                'Excessive_surgical_site_bleeding',
                'Hypotension', 'Tachycardia',
                'Decreased_hemoglobin_level',
                'Signs_of_shock'
            ],
            'Myocardial_Infarction': [
                'Chest_pain_radiating_to_arm_or_jaw',
                'Diaphoresis_excessive_sweating',
                'Shortness_of_breath',
                'Anxiety', 'Nausea_MI'
            ],
            'Thromboembolism': [
                'Chest_pain', 'Sudden_shortness_of_breath',
                'Cyanosis', 'Cold_extremities_Limb_pain',
                'Weak_or_absent_pulse'
            ],
            'Cardiac_Arrhythmias': [
                'Palpitations', 'Dizziness',
                'Syncope_fainting',
                'Fatigue',
                'Shortness_of_breath_Arrhythmia'
            ],
            'Heart_Failure': [
                'Shortness_of_breath_dyspnea',
                'Peripheral_edema_swelling_of_feet_legs',
                'Severe_fatigue',
                'Rapid_weight_gain',
                'Decreased_urine_output_hf'
            ],
            'Wound_Infection': [
                'Fever_Wound',
                'Redness_around_the_wound',
                'Purulent_pus_discharge',
                'Localized_pain',
                'Swelling'
            ]
        }

    # ----------------------------------
    # Core Logic
    # ----------------------------------
    def predict(self, signs_dict):
        active_signs = [k for k, v in signs_dict.items() if v]
        active_count = len(active_signs)

        if not active_signs:
            return self._empty_result()

        scores = {}
        for comp, signs in self.complications.items():
            matched = sum(1 for s in active_signs if s in signs)
            total = len(signs)
            confidence = round(matched / total, 3) if total else 0

            scores[comp] = {
                "matching_signs": matched,
                "total_signs": total,
                "confidence": confidence,
                "priority": self.medical_priority.get(comp, 99)
            }

        max_match = max(v["matching_signs"] for v in scores.values())
        if max_match == 0:
            return self._unknown_result(active_count)

        # Find top complication(s)
        top = [c for c, v in scores.items() if v["matching_signs"] == max_match]
        prediction = min(top, key=lambda c: scores[c]["priority"])

        # Calculate certainty for main prediction
        certainty = self._calculate_certainty(prediction, scores)
        confidence = scores[prediction]["confidence"]

        return {
            "prediction": prediction,
            "confidence": confidence,
            "confidence_level": self._confidence_level(confidence),
            "certainty": certainty,
            "certainty_level": self._certainty_level(certainty),
            "risk_level": self._risk_level(confidence, certainty),
            "decision_layer": "Highest Matching Signs" if len(top) == 1 else f"Medical Priority (tie {len(top)})",
            "matching_signs": scores[prediction]["matching_signs"],
            "total_signs": scores[prediction]["total_signs"],
            "active_signs_count": active_count,
            "priority": scores[prediction]["priority"],
            "all_complications": self._build_list(scores, active_count)
        }

    # ----------------------------------
    # Certainty Calculation
    # ----------------------------------
    def _calculate_certainty(self, comp, scores):
        """Calculate certainty score for a complication"""
        main = scores[comp]['confidence']
        others = [v['confidence'] for k, v in scores.items() if k != comp]
        separation = main - max(others, default=0)
        
        certainty = main * 100 + separation * 40 + min(scores[comp]['matching_signs'] / 3, 1) * 20
        return round(max(10, min(certainty, 100)), 1)

    # ----------------------------------
    # Helpers
    # ----------------------------------
    def _build_list(self, scores, active_count):
        result = []
        for comp, data in scores.items():
            if data["matching_signs"] > 0:
                confidence = data["confidence"]
                certainty = self._calculate_certainty(comp, scores)
                
                result.append({
                    "prediction": comp,
                    "confidence": confidence,
                    "confidence_level": self._confidence_level(confidence),
                    "certainty": certainty,
                    "certainty_level": self._certainty_level(certainty),
                    "risk_level": self._risk_level(confidence, certainty),
                    "decision_layer": "Alternative",
                    "matching_signs": data["matching_signs"],
                    "total_signs": data["total_signs"],
                    "active_signs_count": active_count,
                    "priority": data["priority"]
                })

        return sorted(
            result,
            key=lambda x: (-x["matching_signs"], x["priority"])
        )

    def _confidence_level(self, c):
        if c >= 0.8: return "Very High"
        if c >= 0.65: return "High"
        if c >= 0.5: return "Moderate"
        return "Low"

    def _certainty_level(self, c):
        if c >= 85: return "Very Certain"
        if c >= 70: return "Certain"
        if c >= 50: return "Moderately Certain"
        if c >= 30: return "Uncertain"
        return "Very Uncertain"

    def _risk_level(self, conf, cert):
        if conf >= 0.65 and cert >= 70: return "HIGH"
        if conf >= 0.5 and cert >= 50: return "MODERATE"
        if conf >= 0.3 or cert >= 30: return "LOW"
        return "MINIMAL"

    def _empty_result(self):
        return {
            "prediction": "No_Complication",
            "confidence": 1.0,
            "confidence_level": "Very High",
            "certainty": 100.0,
            "certainty_level": "Very Certain",
            "risk_level": "MINIMAL",
            "decision_layer": "No Signs",
            "matching_signs": 0,
            "total_signs": 0,
            "active_signs_count": 0,
            "priority": 0,
            "all_complications": []
        }

    def _unknown_result(self, active_count):
        return {
            "prediction": "Unknown_Pattern",
            "confidence": 0.0,
            "confidence_level": "Low",
            "certainty": 0.0,
            "certainty_level": "Very Uncertain",
            "risk_level": "MINIMAL",
            "decision_layer": "No Match",
            "matching_signs": 0,
            "total_signs": 0,
            "active_signs_count": active_count,
            "priority": 99,
            "all_complications": []
        }


# ======================================
# Flask App (WSGI)
# ======================================
def create_app():
    app = Flask(__name__)
    CORS(app)

    predictor = CardiacComplicationPredictor()

    @app.route("/api/predict", methods=["POST"])
    def predict():
        data = request.get_json()
        return jsonify({
            "success": True,
            "data": predictor.predict(data.get("signs", {}))
        })

    return app