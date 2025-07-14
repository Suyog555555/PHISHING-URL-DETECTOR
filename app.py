from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from utils.url_features import URLFeatureExtractor

app = Flask(__name__)

# Load the trained model
model_data = None
model = None
extractor = None

def load_model():
    global model_data, model, extractor
    try:
        model_data = joblib.load('phishing_model.pkl')
        model = model_data['model']
        extractor = model_data['extractor']
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def create_default_model():
    """Create a simple default model if trained model is not available"""
    global model, extractor
    print("Creating default model...")
    
    # Import here to avoid circular imports
    from train_model import train_model
    
    # Train and save the model
    model, extractor, accuracy = train_model()
    return True

# Initialize model
if not load_model():
    print("Trained model not found. Training new model...")
    create_default_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get URL from request
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Extract features
        features = extractor.extract_features(url)
        features_array = np.array(list(features.values())).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Calculate confidence
        confidence = probability[1] if prediction == 1 else probability[0]
        
        # Prepare result
        result = {
            'url': url,
            'prediction': 'phishing' if prediction == 1 else 'legitimate',
            'confidence': float(confidence),
            'risk_level': get_risk_level(confidence, prediction),
            'features': features
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_risk_level(confidence, prediction):
    """Determine risk level based on confidence and prediction"""
    if prediction == 1:  # Phishing
        if confidence > 0.8:
            return 'HIGH'
        elif confidence > 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    else:  # Legitimate
        if confidence > 0.8:
            return 'SAFE'
        elif confidence > 0.6:
            return 'LIKELY_SAFE'
        else:
            return 'UNCERTAIN'

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple URLs at once"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({'error': 'No URLs provided'}), 400
        
        results = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
                
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            # Extract features and predict
            features = extractor.extract_features(url)
            features_array = np.array(list(features.values())).reshape(1, -1)
            
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0]
            confidence = probability[1] if prediction == 1 else probability[0]
            
            results.append({
                'url': url,
                'prediction': 'phishing' if prediction == 1 else 'legitimate',
                'confidence': float(confidence),
                'risk_level': get_risk_level(confidence, prediction)
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Return information about the loaded model"""
    try:
        if model_data:
            return jsonify({
                'accuracy': model_data.get('accuracy', 'Unknown'),
                'feature_count': len(model_data.get('feature_names', [])),
                'model_type': 'Random Forest Classifier',
                'features': model_data.get('feature_names', [])
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'extractor_loaded': extractor is not None
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('utils', exist_ok=True)
    
    # Create __init__.py for utils package
    with open('utils/__init__.py', 'w') as f:
        f.write('')
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)