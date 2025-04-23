import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier  # or any model you used
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ✅ 1. Load the dataset
df = pd.read_csv(r'D:\new project titils and documentation pdf\AI-Based Phishing Website Detection\Phishing-Website-Detection-main\Phishing_Legitimate_full.csv')
print(df.head())
print(df.columns) 