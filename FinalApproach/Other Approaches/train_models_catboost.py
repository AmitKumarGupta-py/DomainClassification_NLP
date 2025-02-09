import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# Load dataset
df = pd.read_csv("final_dataset.csv")

def preprocess_text(text):
    """Clean text by lowercasing and removing non-alphanumeric characters."""
    text = text.lower()
    return text

df["cleaned_text"] = df["description"].apply(preprocess_text)

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df["cleaned_text"])
with open("artifacts/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# Label Encoding
domain_encoder = LabelEncoder()
sub_domain_encoder = LabelEncoder()
domain_encoded = domain_encoder.fit_transform(df["domain"])
sub_domain_encoded = sub_domain_encoder.fit_transform(df["sub_domain"])

with open("artifacts/domain_label_encoder.pkl", "wb") as f:
    pickle.dump(domain_encoder, f)
with open("artifacts/sub_domain_label_encoder.pkl", "wb") as f:
    pickle.dump(sub_domain_encoder, f)

# Split Data
X_train, X_test, y_train_domain, y_test_domain, y_train_sub_domain, y_test_sub_domain = train_test_split(
    X_tfidf, domain_encoded, sub_domain_encoded, test_size=0.2, random_state=42
)

# Dictionary to store performance metrics
model_metrics = {
    "catboost_domain": {},
    "catboost_sub_domain": {}
}

# Train CatBoost for Domain Prediction
catboost_domain = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, loss_function='MultiClass', verbose=100)
catboost_domain.fit(X_train, y_train_domain)
y_pred = catboost_domain.predict(X_test)
acc = accuracy_score(y_test_domain, y_pred)
f1 = f1_score(y_test_domain, y_pred, average="weighted")
model_metrics["catboost_domain"] = {"accuracy": acc, "f1": f1}
with open("models/domain_CatBoost.pkl", "wb") as f:
    pickle.dump(catboost_domain, f)

# Train CatBoost for Sub-Domain Prediction
catboost_sub_domain = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, loss_function='MultiClass', verbose=100)
catboost_sub_domain.fit(X_train, y_train_sub_domain)
y_pred = catboost_sub_domain.predict(X_test)
acc = accuracy_score(y_test_sub_domain, y_pred)
f1 = f1_score(y_test_sub_domain, y_pred, average="weighted")
model_metrics["catboost_sub_domain"] = {"accuracy": acc, "f1": f1}
with open("models/sub_domain_CatBoost.pkl", "wb") as f:
    pickle.dump(catboost_sub_domain, f)

# Save model metrics
with open("artifacts/model_metrics.pkl", "wb") as f:
    pickle.dump(model_metrics, f)

print("CatBoost models trained and saved successfully.")
