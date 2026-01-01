"""
========================================
CARDIAC SURGERY COMPLICATION PREDICTION SYSTEM
Simple Confidence Calculation: matching_signs / total_signs
========================================
"""

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ================================
# Predictor Class
# ================================
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
            'Delirium': ['Disorientation', 'Agitation', 'Altered_level_of_consciousness', 'Restlessness', 'Visual'],
            'Acute_Kidney_Injury': ['Decreased_urine_output_Aki', 'Fluid_retention', 'Elevated_creatinine', 'High_BP', 'Swelling'],
            'Atelectasis_Pneumonia': ['Dyspnea', 'Decreased_oxygen_saturation', 'Crackles', 'Cough_atelectasis', 'Increased_body_temperature'],
            'Hemorrhage_Bleeding': ['Excessive_surgical_site_bleeding', 'Hypotension', 'Tachycardia', 'Decreased_hemoglobin_level', 'Signs_of_shock'],
            'Myocardial_Infarction': ['Chest_pain_radiating_to_arm_or_jaw', 'Diaphoresis_excessive_sweating', 'Shortness_of_breath', 'Anxiety', 'Nausea_MI'],
            'Thromboembolism': ['Chest_pain', 'Sudden_shortness_of_breath', 'Cyanosis', 'Cold_extremities_Limb_pain', 'Weak_or_absent_pulse'],
            'Cardiac_Arrhythmias': ['Palpitations', 'Dizziness', 'Syncope_fainting', 'Fatigue', 'Shortness_of_breath_Arrhythmia'],
            'Heart_Failure': ['Shortness_of_breath_dyspnea', 'Peripheral_edema_swelling_of_feet_legs', 'Severe_fatigue', 'Rapid_weight_gain', 'Decreased_urine_output_hf'],
            'Wound_Infection': ['Fever_Wound', 'Redness_around_the_wound', 'Purulent_pus_discharge', 'Localized_pain', 'Inflammation']
        }

    def _simple_confidence(self, matching, total):
        """Calculate simple confidence: matching / total"""
        if total == 0:
            return 0.0
        return round(matching / total, 3)

    def _calculate_certainty(self, comp, scores):
        """Calculate certainty score for a complication"""
        main = scores[comp]['confidence']
        others = [v['confidence'] for k, v in scores.items() if k != comp]
        separation = main - max(others, default=0)
        
        certainty = main * 100 + separation * 40 + min(scores[comp]['score'] / 3, 1) * 20
        return round(max(10, min(certainty, 100)), 1)

    def _confidence_level(self, c):
        """Get confidence level label"""
        if c >= 0.8: return "Very High"
        if c >= 0.65: return "High"
        if c >= 0.5: return "Moderate"
        return "Low"

    def _certainty_level(self, c):
        """Get certainty level label"""
        if c >= 85: return "Very Certain"
        if c >= 70: return "Certain"
        if c >= 50: return "Moderately Certain"
        if c >= 30: return "Uncertain"
        return "Very Uncertain"

    def _risk_level(self, conf, cert):
        """Calculate risk level based on confidence and certainty"""
        if conf >= 0.65 and cert >= 70: return "HIGH"
        if conf >= 0.5 and cert >= 50: return "MODERATE"
        if conf >= 0.3 or cert >= 30: return "LOW"
        return "MINIMAL"

    def _calculate_all_scores(self, active_signs):
        """Calculate scores for all complications"""
        scores = {}
        active_count = len(active_signs)
        
        for comp, signs in self.complications.items():
            matching_count = sum(1 for s in active_signs if s in signs)
            confidence = self._simple_confidence(matching_count, len(signs))
            
            scores[comp] = {
                "score": matching_count,
                "confidence": confidence,
                "total_signs": len(signs),
                "priority": self.medical_priority.get(comp, 99)
            }
        return scores

    def _build_complication_result(self, comp_name, comp_data, scores, active_count, max_score):
        """Build result dictionary for a single complication"""
        confidence = comp_data['confidence']
        certainty = self._calculate_certainty(comp_name, scores)
        match_percentage = round((comp_data['score'] / comp_data['total_signs']) * 100, 1)
        
        # Determine decision layer
        top = [c for c, v in scores.items() if v['score'] == max_score]
        if len(top) == 1 and comp_name == top[0]:
            decision_layer = "Layer 2: Highest Score"
        elif comp_name in top:
            decision_layer = f"Layer 3: Medical Priority (tie {len(top)})"
        else:
            decision_layer = f"Layer 1: Score {comp_data['score']}"
        
        return {
            "prediction": comp_name,
            "confidence": confidence,
            "confidence_level": self._confidence_level(confidence),
            "certainty": certainty,
            "certainty_level": self._certainty_level(certainty),
            "risk_level": self._risk_level(confidence, certainty),
            "decision_layer": decision_layer,
            "matching_signs": comp_data['score'],
            "total_signs": comp_data['total_signs'],
            "match_percentage": match_percentage,
            "active_signs_count": active_count,
            "priority": comp_data['priority']
        }

    def _all_complications(self, scores, active_count):
        """Generate list of all complications with matching signs"""
        max_score = max(v['score'] for v in scores.values())
        result = [
            self._build_complication_result(comp_name, comp_data, scores, active_count, max_score)
            for comp_name, comp_data in scores.items()
            if comp_data['score'] > 0
        ]
        
        # Sort by matching_signs (desc), then medical priority (asc)
        return sorted(result, key=lambda x: (-x['matching_signs'], x['priority']))

    def predict_with_logic(self, signs_dict):
        """Main prediction logic"""
        active_signs = [s for s, v in signs_dict.items() if v]

        if not active_signs:
            return {
                "prediction": "No_Complication",
                "confidence": 1.0,
                "confidence_level": "Very High",
                "certainty": 100.0,
                "certainty_level": "Very Certain",
                "risk_level": "MINIMAL",
                "decision_layer": "No Signs Detected",
                "matching_signs": 0,
                "total_signs": 0,
                "match_percentage": 0.0,
                "active_signs_count": 0,
                "priority": 0,
                "all_complications": []
            }

        scores = self._calculate_all_scores(active_signs)
        max_score = max(v['score'] for v in scores.values())
        
        if max_score == 0:
            return {
                "prediction": "Unknown_Pattern",
                "confidence": 0.0,
                "confidence_level": "Low",
                "certainty": 0.0,
                "certainty_level": "Very Uncertain",
                "risk_level": "MINIMAL",
                "decision_layer": "No Match Found",
                "matching_signs": 0,
                "total_signs": 0,
                "match_percentage": 0.0,
                "active_signs_count": len(active_signs),
                "priority": 99,
                "all_complications": []
            }

        # Find top complication(s)
        top = [c for c, v in scores.items() if v['score'] == max_score]
        
        if len(top) == 1:
            comp = top[0]
            layer = "Layer 2: Highest Score"
        else:
            comp = min(top, key=lambda c: self.medical_priority.get(c, 99))
            layer = f"Layer 3: Medical Priority (tie {len(top)})"

        # Build main result
        comp_data = scores[comp]
        certainty = self._calculate_certainty(comp, scores)
        confidence = comp_data['confidence']
        match_percentage = round((comp_data['score'] / comp_data['total_signs']) * 100, 1)

        return {
            "prediction": comp,
            "confidence": confidence,
            "confidence_level": self._confidence_level(confidence),
            "certainty": certainty,
            "certainty_level": self._certainty_level(certainty),
            "risk_level": self._risk_level(confidence, certainty),
            "decision_layer": layer,
            "matching_signs": comp_data['score'],
            "total_signs": comp_data['total_signs'],
            "match_percentage": match_percentage,
            "active_signs_count": len(active_signs),
            "priority": comp_data['priority'],
            "all_complications": self._all_complications(scores, len(active_signs))
        }


# ================================
# Flask App
# ================================
def create_app():
    app = Flask(__name__)
    CORS(app)
    predictor = CardiacComplicationPredictor()

    @app.route('/api/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        return jsonify({
            "success": True,
            "data": predictor.predict_with_logic(data['signs'])
        })

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)