import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier  # or any model you used
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ✅ 1. Load the dataset
df = pd.read_csv(r'D:\new project titils and documentation pdf\AI-Based Phishing Website Detection\Phishing-Website-Detection-main\Phishing_Legitimate_full.csv')

# ✅ 2. Drop the 'id' column
df.drop(columns=['id'], inplace=True)

# ✅ 3. Check for missing values
print("🔍 Missing values in dataset:", df.isnull().sum().sum())

# ✅ 4. Split features and labels
X = df.drop('CLASS_LABEL', axis=1)
y = df['CLASS_LABEL']

# ✅ 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 6. Train the model (RandomForest is just an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ 7. Save the model
joblib.dump(model, 'phishing_model.pkl')
print("💾 Model saved successfully as phishing_model.pkl")

# ✅ 8. Load the model
loaded_model = joblib.load('phishing_model.pkl')
print("📦 Model loaded successfully.")

# ✅ 9. Predict using the loaded model
y_pred = loaded_model.predict(X_test)

# ✅ 10. Evaluate performance
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ 11. Optional: Predict on a single sample
sample_input = X_test.iloc[0].values.reshape(1, -1)
sample_actual = y_test.iloc[0]
sample_prediction = loaded_model.predict(sample_input)[0]

print("\n🔍 Single Sample Test")
print("Predicted Label:", sample_prediction)
print("Actual Label   :", sample_actual)
