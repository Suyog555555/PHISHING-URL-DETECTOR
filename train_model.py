import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from utils.url_features import URLFeatureExtractor

def create_sample_data():
    """Create sample training data"""
    # Sample phishing URLs
    phishing_urls = [
        'http://paypal-update.secure-account.com/login',
        'https://amazon-security.tk/verify-account',
        'http://facebook-security.ml/confirm-identity',
        'https://192.168.1.1/bank-login',
        'http://bit.ly/urgent-update',
        'https://apple-id.secure-verification.ga/update',
        'http://microsoft-security.cf/account-suspended',
        'https://google-account.verify-now.com/signin',
        'http://bank-of-america.update-account.org/login',
        'https://twitter-security.suspicious-domain.tk/verify',
        'http://linkedin-account.confirm-details.ml/update',
        'https://instagram-security.urgent-action.ga/verify',
        'http://netflix-account.renew-subscription.cf/login',
        'https://spotify-premium.activate-account.tk/signin',
        'http://dropbox-storage.upgrade-account.ml/verify',
        'https://github-security.confirm-identity.ga/login',
        'http://adobe-account.update-payment.cf/verify',
        'https://zoom-security.account-suspended.tk/signin',
        'http://whatsapp-web.verify-account.ml/confirm',
        'https://telegram-security.urgent-update.ga/verify'
    ]
    
    # Sample legitimate URLs
    legitimate_urls = [
        'https://www.google.com/search?q=python',
        'https://github.com/user/repository',
        'https://stackoverflow.com/questions/12345',
        'https://www.amazon.com/product/dp/B08N5WRWNW',
        'https://docs.python.org/3/library/urllib.html',
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'https://www.wikipedia.org/wiki/Machine_learning',
        'https://www.reddit.com/r/programming',
        'https://www.linkedin.com/in/profile',
        'https://www.facebook.com/profile',
        'https://www.twitter.com/username',
        'https://www.instagram.com/username',
        'https://www.netflix.com/browse',
        'https://www.spotify.com/playlist/12345',
        'https://www.dropbox.com/s/abc123/file.pdf',
        'https://www.adobe.com/products/photoshop',
        'https://zoom.us/j/1234567890',
        'https://web.whatsapp.com/',
        'https://telegram.org/apps',
        'https://www.microsoft.com/office'
    ]
    
    # Create DataFrame
    urls = phishing_urls + legitimate_urls
    labels = [1] * len(phishing_urls) + [0] * len(legitimate_urls)  # 1 for phishing, 0 for legitimate
    
    df = pd.DataFrame({
        'url': urls,
        'label': labels
    })
    
    return df

def train_model():
    """Train the phishing detection model"""
    print("Creating sample training data...")
    df = create_sample_data()
    
    # Initialize feature extractor
    extractor = URLFeatureExtractor()
    
    print("Extracting features from URLs...")
    # Extract features for all URLs
    features_list = []
    for url in df['url']:
        features = extractor.extract_features(url)
        features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Prepare training data
    X = features_df.values
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Random Forest model...")
    # Train Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Feature importance
    feature_names = extractor.get_feature_names()
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Save the model and feature extractor
    print("\nSaving model and feature extractor...")
    model_data = {
        'model': model,
        'extractor': extractor,
        'feature_names': feature_names,
        'accuracy': accuracy
    }
    
    joblib.dump(model_data, 'phishing_model.pkl')
    print("Model saved as 'phishing_model.pkl'")
    
    # Save sample data for reference
    df.to_csv('data/sample_urls.csv', index=False)
    print("Sample data saved as 'data/sample_urls.csv'")
    
    return model, extractor, accuracy

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Train the model
    model, extractor, accuracy = train_model()
    
    # Test with a few examples
    print("\nTesting model with sample URLs:")
    test_urls = [
        'https://www.google.com',
        'http://paypal-security.tk/login',
        'https://github.com/user/repo',
        'http://192.168.1.1/bank-login'
    ]
    
    for url in test_urls:
        features = extractor.extract_features(url)
        features_array = np.array(list(features.values())).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        result = "PHISHING" if prediction == 1 else "LEGITIMATE"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        print(f"URL: {url}")
        print(f"Prediction: {result} (Confidence: {confidence:.4f})")
        print("-" * 50)