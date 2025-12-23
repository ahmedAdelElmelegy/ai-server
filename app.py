"""
Surgery Complication Prediction API
Train Random Forest model and serve predictions via Flask API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Global variables for model and encoders
model = None
label_encoders = {}
feature_columns = []
target_encoder = None

def create_sample_data():
    """Create sample training data if CSV doesn't exist"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 90, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Surgery_Type': np.random.choice(['Cardiac', 'Orthopedic', 'Abdominal', 'Neurosurgery'], n_samples),
        'Diabetes': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'Hypertension': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
        'ECG_Monitoring': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
        'Early_Ambulation': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'Incentive_Spirometry': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
        'Wound_Care': np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2]),
        'Urine_Output_Monitoring': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
        'Blood_Glucose_Monitoring': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
    }
    
    # Generate complications based on risk factors
    complications = []
    for i in range(n_samples):
        risk = 0
        if data['Age'][i] > 65: risk += 2
        if data['Diabetes'][i] == 'Yes': risk += 2
        if data['Hypertension'][i] == 'Yes': risk += 1
        if data['ECG_Monitoring'][i] == 'No': risk += 1
        if data['Early_Ambulation'][i] == 'No': risk += 1
        if data['Wound_Care'][i] == 'No': risk += 2
        
        rand = np.random.random()
        if risk >= 6 and rand < 0.6:
            complications.append(np.random.choice(['Sepsis', 'AKI', 'Pneumonia', 'SSI']))
        elif risk >= 4 and rand < 0.4:
            complications.append(np.random.choice(['Arrhythmia', 'Hyperglycemia', 'Atelectasis']))
        elif risk >= 2 and rand < 0.2:
            complications.append(np.random.choice(['Stroke', 'Pneumonia', 'No_Complication']))
        else:
            complications.append('No_Complication')
    
    data['Complication'] = complications
    
    df = pd.DataFrame(data)
    df.to_csv('training_data.csv', index=False)
    print("Sample training data created: training_data.csv")
    return df

def train_model():
    """Train Random Forest model on training data"""
    global model, label_encoders, feature_columns, target_encoder
    
    # Load or create training data
    if not os.path.exists('training_data.csv'):
        print("training_data.csv not found. Creating sample data...")
        df = create_sample_data()
    else:
        df = pd.read_csv('training_data.csv')
    
    print(f"Training data shape: {df.shape}")
    print(f"Complication distribution:\n{df['Complication'].value_counts()}")
    
    # Separate features and target
    X = df.drop('Complication', axis=1)
    y = df['Complication']
    
    feature_columns = X.columns.tolist()
    
    # Encode categorical features
    X_encoded = X.copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Testing accuracy: {test_acc:.3f}")
    
    # Save model and encoders
    joblib.dump(model, 'rf_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(target_encoder, 'target_encoder.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    print("Model and encoders saved successfully!")
    
    return model

def load_model():
    """Load trained model and encoders"""
    global model, label_encoders, feature_columns, target_encoder
    
    try:
        model = joblib.load('rf_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        print("Model loaded successfully!")
        return True
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        train_model()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict surgery complication"""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'Age', 'Gender', 'Surgery_Type', 'Diabetes', 'Hypertension',
            'ECG_Monitoring', 'Early_Ambulation', 'Incentive_Spirometry',
            'Wound_Care', 'Urine_Output_Monitoring', 'Blood_Glucose_Monitoring'
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Create DataFrame with input
        input_df = pd.DataFrame([data])
        
        # Reorder columns to match training data
        input_df = input_df[feature_columns]
        
        # Encode categorical features
        input_encoded = input_df.copy()
        for col in input_df.columns:
            if col in label_encoders:
                try:
                    input_encoded[col] = label_encoders[col].transform(input_df[col])
                except ValueError as e:
                    return jsonify({
                        'error': f'Invalid value for {col}: {input_df[col].values[0]}'
                    }), 400
        
        # Make prediction
        prediction_encoded = model.predict(input_encoded)[0]
        prediction_proba = model.predict_proba(input_encoded)[0]
        
        # Decode prediction
        prediction = target_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence scores for all classes
        confidence_scores = {
            target_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(prediction_proba)
        }
        
        # Sort by probability
        sorted_predictions = sorted(
            confidence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(prediction_proba.max()),
            'all_probabilities': confidence_scores,
            'top_predictions': [
                {'complication': comp, 'probability': prob}
                for comp, prob in sorted_predictions[:3]
            ]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model"""
    try:
        train_model()
        return jsonify({'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Initializing Surgery Complication Prediction API...")
    load_model()
    print("Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)