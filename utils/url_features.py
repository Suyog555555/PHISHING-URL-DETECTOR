import re
import urllib.parse
from urllib.parse import urlparse

class URLFeatureExtractor:
    def __init__(self):
        # Common URL shorteners
        self.shorteners = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'short.link',
            'ow.ly', 'buff.ly', 'adf.ly', 'tiny.cc', 'lnkd.in'
        ]
        
        # Suspicious keywords often found in phishing URLs
        self.suspicious_keywords = [
            'secure', 'account', 'update', 'confirm', 'verify', 'login',
            'signin', 'bank', 'paypal', 'amazon', 'apple', 'microsoft',
            'google', 'facebook', 'twitter', 'suspend', 'limited',
            'expire', 'click', 'here', 'now', 'urgent', 'immediately'
        ]
    
    def extract_features(self, url):
        """Extract features from a URL for phishing detection"""
        try:
            # Parse URL
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            query = parsed.query.lower()
            
            features = {}
            
            # Basic URL features
            features['url_length'] = len(url)
            features['domain_length'] = len(domain)
            features['path_length'] = len(path)
            features['query_length'] = len(query)
            
            # Character count features
            features['dot_count'] = url.count('.')
            features['hyphen_count'] = url.count('-')
            features['underscore_count'] = url.count('_')
            features['slash_count'] = url.count('/')
            features['question_count'] = url.count('?')
            features['equal_count'] = url.count('=')
            features['and_count'] = url.count('&')
            features['digit_count'] = sum(c.isdigit() for c in url)
            
            # Special character ratio
            special_chars = sum(not c.isalnum() for c in url)
            features['special_char_ratio'] = special_chars / len(url) if len(url) > 0 else 0
            
            # Domain features
            features['subdomain_count'] = len(domain.split('.')) - 2 if domain else 0
            features['domain_has_hyphen'] = 1 if '-' in domain else 0
            
            # Protocol features
            features['is_https'] = 1 if parsed.scheme == 'https' else 0
            features['has_port'] = 1 if ':' in domain and not domain.startswith('www.') else 0
            
            # IP address detection
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            features['has_ip'] = 1 if re.search(ip_pattern, domain) else 0
            
            # URL shortener detection
            features['is_shortened'] = 1 if any(short in domain for short in self.shorteners) else 0
            
            # Suspicious keyword count
            full_url_lower = url.lower()
            features['suspicious_word_count'] = sum(1 for word in self.suspicious_keywords if word in full_url_lower)
            
            # Suspicious patterns
            features['has_suspicious_tld'] = 1 if any(tld in domain for tld in ['.tk', '.ml', '.ga', '.cf']) else 0
            features['has_multiple_subdomains'] = 1 if features['subdomain_count'] > 2 else 0
            features['has_suspicious_path'] = 1 if any(word in path for word in ['admin', 'login', 'signin', 'account']) else 0
            
            # Length-based features
            features['long_url'] = 1 if len(url) > 100 else 0
            features['long_domain'] = 1 if len(domain) > 30 else 0
            
            # Hexadecimal pattern in URL
            hex_pattern = r'[0-9a-fA-F]{8,}'
            features['has_hex_pattern'] = 1 if re.search(hex_pattern, url) else 0
            
            # Double slash after protocol
            features['has_double_slash'] = 1 if '//' in url[8:] else 0  # Skip protocol part
            
            # Suspicious file extensions
            suspicious_extensions = ['.exe', '.zip', '.rar', '.bat', '.scr', '.pif', '.com']
            features['has_suspicious_extension'] = 1 if any(ext in url.lower() for ext in suspicious_extensions) else 0
            
            return features
            
        except Exception as e:
            # Return default features if URL parsing fails
            return self._get_default_features()
    
    def _get_default_features(self):
        """Return default feature values"""
        return {
            'url_length': 0,
            'domain_length': 0,
            'path_length': 0,
            'query_length': 0,
            'dot_count': 0,
            'hyphen_count': 0,
            'underscore_count': 0,
            'slash_count': 0,
            'question_count': 0,
            'equal_count': 0,
            'and_count': 0,
            'digit_count': 0,
            'special_char_ratio': 0,
            'subdomain_count': 0,
            'domain_has_hyphen': 0,
            'is_https': 0,
            'has_port': 0,
            'has_ip': 0,
            'is_shortened': 0,
            'suspicious_word_count': 0,
            'has_suspicious_tld': 0,
            'has_multiple_subdomains': 0,
            'has_suspicious_path': 0,
            'long_url': 0,
            'long_domain': 0,
            'has_hex_pattern': 0,
            'has_double_slash': 0,
            'has_suspicious_extension': 0
        }
    
    def get_feature_names(self):
        """Return list of feature names"""
        return list(self._get_default_features().keys())