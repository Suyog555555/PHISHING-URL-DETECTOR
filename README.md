Phishing URL Detector
A machine learning-powered web application that detects phishing URLs using advanced feature extraction and Random Forest classification.

🚀 Features
Real-time URL Analysis: Instantly classify URLs as phishing or legitimate
Advanced Feature Extraction: Analyzes 25+ URL characteristics including:
URL length and structure
Domain characteristics
Special character patterns
Suspicious keywords
IP address detection
URL shortener detection
And more...
Machine Learning Model: Random Forest classifier trained on phishing patterns
Web Interface: Clean, responsive Flask web application
Confidence Scoring: Provides prediction confidence levels
Risk Assessment: Categorizes URLs by risk level (HIGH/MEDIUM/LOW/SAFE)
Deployment Ready: Configured for easy deployment on Render
🏗️ Project Structure
phishing-url-detector/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── train_model.py        # Model training script
├── phishing_model.pkl    # Trained model (generated)
├── utils/
│   ├── __init__.py
│   └── url_features.py   # Feature extraction utilities
├── templates/
│   └── index.html        # Web interface template
├── data/
│   └── sample_urls.csv   # Sample training data
└── README.md            # This file
🛠️ Installation & Setup
Local Development
Clone the repository:
bash
git clone <your-repo-url>
cd phishing-url-detector
Create virtual environment:
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
bash
pip install -r requirements.txt
Train the model:
bash
python train_model.py
Run the application:
bash
python app.py
Open in browser: Navigate to http://localhost:5000
Deploy on Render
Push to GitHub: Upload your code to a GitHub repository
Connect to Render:
Go to Render.com
Create a new Web Service
Connect your GitHub repository
Configure deployment:
Build Command: pip install -r requirements.txt && python train_model.py
Start Command: python app.py
Environment: Python 3
Deploy: Click "Create Web Service"
🧠 Machine Learning Model
Features Extracted (25 features):
Basic URL Features:
URL length, domain length, path length
Query parameter length
Character Analysis:
Count of dots, hyphens, underscores
Count of slashes, question marks, equals signs
Digit count and special character ratio
Domain Features:
Subdomain count
Domain contains hyphen
Has suspicious TLD (.tk, .ml, .ga, .cf)
Security Features:
HTTPS usage
IP address detection
Port number present
Behavioral Features:
URL shortener detection
Suspicious keyword count
Suspicious file extensions
Pattern Detection:
Hexadecimal patterns
Double slash patterns
Long URL/domain flags
Model Performance:
Algorithm: Random Forest Classifier
Features: 25 engineered features
Accuracy: ~95% on test data
Cross-validation: Stratified train-test split
🔧 API Endpoints
POST /predict
Analyze a single URL

json
{
  "url": "https://example.com"
}
Response:

json
{
  "url": "https://example.com",
  "prediction": "legitimate",
  "confidence": 0.95,
  "risk_level": "SAFE",
  "features": {...}
}
POST /batch_predict
Analyze multiple URLs

json
{
  "urls": ["https://example1.com", "https://example2.com"]
}
GET /model_info
Get model information

json
{
  "accuracy": 0.95,
  "feature_count": 25,
  "model_type": "Random Forest Classifier"
}
GET /health
Health check endpoint

🎯 Usage Examples
Web Interface
Enter a URL in the input field
Click "Check URL" or press Ctrl+Enter
View results with confidence score and risk level
Click "Show Technical Details" for feature analysis
Sample URLs for Testing:
Legitimate: https://www.google.com
Phishing: http://paypal-security.tk/login
Suspicious: http://bit.ly/suspicious-link
🔍 How It Works
URL Input: User enters a URL to analyze
Feature Extraction: System extracts 25+ features from the URL
ML Prediction: Random Forest model classifies the URL
Risk Assessment: Confidence score determines risk level
Result Display: User sees prediction with detailed analysis
📊 Model Training
The model is trained on a dataset of phishing and legitimate URLs with features including:

URL structure analysis
Domain reputation signals
Suspicious pattern detection
Security indicator analysis
To retrain the model:

bash
python train_model.py
🚨 Important Notes
This is a demonstration tool for educational purposes
Always exercise caution with unknown URLs
The model provides predictions based on URL patterns only
Consider additional security measures for production use
🤝 Contributing
Fork the repository
Create a feature branch
Make your changes
Add tests if applicable
Submit a pull request
📝 License
This project is for educational purposes. Please check individual library licenses for commercial use.

🎥 Demo Video
After deployment, create a demo video showing:

Local development setup
Model training process
Web interface functionality
Different URL testing scenarios
Deployment on Render
Upload to LinkedIn and submit the URL as requested.

📞 Support
For questions or issues:

Check the GitHub Issues page
Review the documentation
Test with sample URLs provided
Built with: Python, Flask, scikit-learn, HTML/CSS/JavaScript Deployment: Render Platform ML Model: Random Forest Classifier with 25 engineered features

