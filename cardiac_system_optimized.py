"""
========================================
CARDIAC SURGERY COMPLICATION PREDICTION SYSTEM
Optimized with 3-Layer Decision Logic
========================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

class CardiacComplicationPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
        # Medical priority (1 = highest risk, 9 = lowest)
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
        
        # Complete medical signs per complication
        self.complications = {
            'Delirium': [
                'Disorientation',
                'Agitation',
                'Altered_level_of_consciousness'
            ],
            'Acute_Kidney_Injury': [
                'Decreased_urine_output_Aki',
                'Fluid_retention',
                'Elevated_creatinine',
                'High_BP',
                'Nausea_and_vomiting'
            ],
            'Atelectasis_Pneumonia': [
                'Dyspnea',
                'Decreased_oxygen_saturation',
                'Crackles',
            ],
            'Hemorrhage_Bleeding': [
                'Excessive_surgical_site_bleeding',
                'Hypotension',
                'Tachycardia',
                'Decreased_hemoglobin_level',
                'Signs_of_shock'
            ],
            'Myocardial_Infarction': [
                'Chest_pain_radiating_to_arm_or_jaw',
                'Diaphoresis_excessive_sweating',
                'Shortness_of_breath',
                'Anxiety'
            ],
            'Thromboembolism': [
                'Chest_pain',
                'Sudden_shortness_of_breath',
                'Cyanosis',
                'Cold_extremities_Limb_pain',
                'Weak_or_absent_pulse'
            ],
            'Cardiac_Arrhythmias': [
                'Palpitations',
                'Dizziness',
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
    
    def predict_with_logic(self, signs_dict):
        """
        3-Layer Decision Logic:
        Layer 1: Exact Match (if all signs belong to one complication only)
        Layer 2: Scoring (count matching signs per complication)
        Layer 3: Priority (use medical priority as tiebreaker)
        """
        active_signs = [sign for sign, is_active in signs_dict.items() if is_active]
        
        if not active_signs:
            return {
                "prediction": "No_Complication",
                "confidence": 1.0,
                "confidence_level": "Very High",
                "decision_layer": "No Signs Detected",
                "score": 0,
                "matching_signs": 0,
                "total_signs": 0
            }
        
        # === LAYER 1: EXACT MATCH ===
        possible_complications = set()
        for sign in active_signs:
            for complication, signs_list in self.complications.items():
                if sign in signs_list:
                    possible_complications.add(complication)
        
        if len(possible_complications) == 1:
            complication = list(possible_complications)[0]
            matching = len([s for s in active_signs if s in self.complications[complication]])
            total = len(self.complications[complication])
            confidence = matching / total if total > 0 else 0
            
            return {
                "prediction": complication,
                "confidence": confidence,
                "confidence_level": self._get_confidence_level(confidence),
                "decision_layer": "Layer 1: Exact Match",
                "score": matching,
                "matching_signs": matching,
                "total_signs": total
            }
        
        # === LAYER 2: SCORING ===
        scores = {}
        for complication, signs_list in self.complications.items():
            matching_signs = [s for s in active_signs if s in signs_list]
            scores[complication] = {
                'score': len(matching_signs),
                'confidence': len(matching_signs) / len(signs_list) if signs_list else 0,
                'matching_signs': matching_signs,
                'total_signs': len(signs_list)
            }
        
        # Get max score
        max_score = max(s['score'] for s in scores.values())
        
        if max_score == 0:
            return {
                "prediction": "Unknown_Pattern",
                "confidence": 0.0,
                "confidence_level": "Low",
                "decision_layer": "No Match Found",
                "score": 0,
                "matching_signs": 0,
                "total_signs": 0
            }
        
        # Get all complications with max score
        top_complications = [
            comp for comp, data in scores.items() 
            if data['score'] == max_score
        ]
        
        if len(top_complications) == 1:
            complication = top_complications[0]
            data = scores[complication]
            return {
                "prediction": complication,
                "confidence": data['confidence'],
                "confidence_level": self._get_confidence_level(data['confidence']),
                "decision_layer": "Layer 2: Highest Score",
                "score": data['score'],
                "matching_signs": data['score'],
                "total_signs": data['total_signs']
            }
        
        # === LAYER 3: PRIORITY (TIEBREAKER) ===
        best_complication = min(
            top_complications,
            key=lambda c: self.medical_priority.get(c, 99)
        )
        
        data = scores[best_complication]
        return {
            "prediction": best_complication,
            "confidence": data['confidence'],
            "confidence_level": self._get_confidence_level(data['confidence']),
            "decision_layer": f"Layer 3: Medical Priority (tied with {len(top_complications)} complications)",
            "score": data['score'],
            "matching_signs": data['score'],
            "total_signs": data['total_signs']
        }
    
    def _get_confidence_level(self, confidence):
        """Determine confidence level"""
        if confidence >= 0.80:
            return "Very High"
        elif confidence >= 0.65:
            return "High"
        elif confidence >= 0.50:
            return "Moderate"
        else:
            return "Low"
    
    def generate_synthetic_data(self, n_samples=5000):
        """Generate synthetic training data"""
        all_signs = sorted(list(set(sign for signs in self.complications.values() 
                                   for sign in signs)))
        self.feature_names = all_signs
        
        data = []
        labels = []
        
        for complication, primary_signs in self.complications.items():
            samples_per_complication = n_samples // len(self.complications)
            
            for _ in range(samples_per_complication):
                sample = {sign: False for sign in all_signs}
                
                # High probability for primary signs (85%)
                for sign in primary_signs:
                    sample[sign] = np.random.random() < 0.85
                
                # Low probability for other signs (15% noise)
                for sign in all_signs:
                    if sign not in primary_signs:
                        sample[sign] = np.random.random() < 0.15
                
                data.append(list(sample.values()))
                labels.append(complication)
        
        return np.array(data), np.array(labels)
    
    def train(self):
        """Train the Random Forest model"""
        print("Generating synthetic training data...")
        X, y = self.generate_synthetic_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Total features (signs): {len(self.feature_names)}")
        print("\nTraining Random Forest model...")
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluation
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        accuracy = (y_pred == y_test).mean()
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        
        return self.model
    
    def save_model(self, filepath='cardiac_model_optimized.pkl'):
        """Save trained model to file"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'complications': self.complications,
            'medical_priority': self.medical_priority
        }, filepath)
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath='cardiac_model_optimized.pkl'):
        """Load trained model from file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.complications = data['complications']
        self.medical_priority = data.get('medical_priority', self.medical_priority)
        print(f"Model loaded from {filepath}")


def train_model():
    """Train and save the model"""
    print("="*60)
    print("CARDIAC COMPLICATION PREDICTION MODEL - TRAINING")
    print("="*60)
    
    predictor = CardiacComplicationPredictor()
    predictor.train()
    predictor.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now run the Flask API:")
    print("  python cardiac_system_optimized.py")
    print("="*60)


def create_app():
    """Create Flask API application"""
    app = Flask(__name__)
    CORS(app)
    
    # Load the trained model
    model_data = joblib.load('cardiac_model_optimized.pkl')
    predictor = CardiacComplicationPredictor()
    predictor.model = model_data['model']
    predictor.feature_names = model_data['feature_names']
    predictor.complications = model_data['complications']
    predictor.medical_priority = model_data.get('medical_priority', predictor.medical_priority)
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'model_loaded': predictor.model is not None,
            'total_features': len(predictor.feature_names),
            'complications': list(predictor.complications.keys()),
            'decision_layers': 3
        })
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        """Predict cardiac complication using 3-layer logic"""
        try:
            data = request.get_json()
            
            if 'signs' not in data:
                return jsonify({'error': 'Missing "signs" field in request body'}), 400
            
            signs_dict = data['signs']
            
            if not isinstance(signs_dict, dict):
                return jsonify({'error': '"signs" must be a dictionary'}), 400
            
            # Use optimized prediction logic
            result = predictor.predict_with_logic(signs_dict)
            
            return jsonify(result), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/features', methods=['GET'])
    def get_features():
        """Get list of all available features/signs"""
        return jsonify({
            'features': predictor.feature_names,
            'count': len(predictor.feature_names)
        })
    
    @app.route('/api/complications', methods=['GET'])
    def get_complications():
        """Get list of all complications with their associated signs"""
        return jsonify({
            'complications': predictor.complications,
            'medical_priority': predictor.medical_priority
        })
    
    return app


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_model()
    else:
        app = create_app()
        # print("\n" + "="*50)
        # print("CARDIAC COMPLICATION PREDICTION API")
        # print("Optimized with 3-Layer Decision Logic")
        # print("="*50)
        # print("\nEndpoints:")
        # print("  GET  /api/health        - Health check")
        # print("  POST /api/predict       - Predict complication")
        # print("  GET  /api/features      - List all signs")
        # print("  GET  /api/complications - Get complications")
        # print("\nAPI running on http://localhost:5000")
        print("="*50 + "\n")
        port = int(os.environ.get("PORT", 5000))  # يستخدم بورت Koyeb لو متاح
        app.run(host="0.0.0.0", port=port)


"""
REQUIREMENTS:
pip install flask flask-cors numpy pandas scikit-learn joblib

USAGE:
1. Train model: python cardiac_system_optimized.py train
2. Run API: python cardiac_system_optimized.py
"""