from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import re
from urllib.parse import urlparse

# Load the trained model and scaler
model_path = './url_classifier_model.joblib'
scaler_path = './scaler.joblib'
model = load(model_path)
scaler = load(scaler_path)

app = Flask(__name__)

def extract_features(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname if parsed_url.hostname else ''
    path = parsed_url.path if parsed_url.path else ''
    
    features = {
        'length_url': len(url),
        'length_hostname': len(hostname),
        'ip': int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', hostname))),
        'nb_dots': url.count('.'),
        'nb_hyphens': url.count('-'),
        'nb_at': url.count('@'),
        'nb_qm': url.count('?'),
        'nb_and': url.count('&'),
        'nb_or': url.count('|'),
        'nb_eq': url.count('='),
        'nb_underscore': url.count('_'),
        'nb_tilde': url.count('~'),
        'nb_percent': url.count('%'),
        'nb_slash': url.count('/'),
        'nb_star': url.count('*'),
        'nb_colon': url.count(':'),
        'nb_comma': url.count(','),
        'nb_semicolon': url.count(';'),
        'nb_dollar': url.count('$'),
        'nb_space': url.count(' '),
        'nb_www': url.count('www'),
        'nb_com': url.count('.com'),
        'nb_dslash': url.count('//'),
        'http_in_path': int('http' in path.lower()),
        'https_token': int('https' in path.lower()),
        'ratio_digits_url': sum(c.isdigit() for c in url) / len(url),
        'ratio_digits_host': sum(c.isdigit() for c in hostname) / len(hostname),
        'nb_redirection': url.count('//'),
        'length_words_raw': len(re.findall(r'\w+', url)),
        'char_repeat': max([len(m.group(0)) for m in re.finditer(r'(.)\1*', url)], default=0),
        'shortest_word_length': min([len(word) for word in re.findall(r'\w+', url)], default=0),
        'longest_word_length': max([len(word) for word in re.findall(r'\w+', url)], default=0),
        'avg_word_length': np.mean([len(word) for word in re.findall(r'\w+', url)]) if re.findall(r'\w+', url) else 0
    }

    feature_order = [
        'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq',
        'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolon', 'nb_dollar',
        'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url', 'ratio_digits_host',
        'nb_redirection', 'length_words_raw', 'char_repeat', 'shortest_word_length', 'longest_word_length', 'avg_word_length'
    ]

    return [features[feature] for feature in feature_order]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    features = extract_features(url)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    output = 'malicious' if prediction[0] == 1 else 'legitimate'
    return jsonify(result=output)

if __name__ == '__main__':
    app.run()
