from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Corrected feature names (49 total)
feature_names = [
    'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
    'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
    'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
    'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
    'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
    'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath',
    'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks',
    'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
    'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
    'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch',
    'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow',
    'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle',
    'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT',
    'PctExtResourceUrlsRT', 'AbnormalExtFormActionR', 'ExtMetaScriptLinkRT',
    'PctExtNullSelfRedirectHyperlinksRT'
]

# ✅ Load model
model = joblib.load('phishing_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input values in correct order
        input_data = [float(request.form.get(f)) for f in feature_names]
        features = np.array(input_data).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
        result = "✅ Legitimate Website" if prediction == 0 else "⚠️ Phishing Website"

        return render_template('index.html', features=feature_names, prediction_text=f'Result: {result}')
    except Exception as e:
        return render_template('index.html', features=feature_names, prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
