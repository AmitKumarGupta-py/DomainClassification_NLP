import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from catboost import CatBoostClassifier

# Set Streamlit Page Config (Must be at the top)
st.set_page_config(page_title="Domain & Sub-Domain Prediction", layout="wide")

# Set Seaborn Style
sns.set(style="whitegrid")

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_text(text):
    """Clean text by lowercasing and removing non-alphanumeric characters."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# -----------------------------
# Load Artifacts and Models (with caching)
# -----------------------------
@st.cache_resource
def load_artifacts():
    """Load and cache all necessary models and vectorizers."""
    artifacts = {}

    # Load TF-IDF vectorizer
    with open("artifacts/tfidf_vectorizer.pkl", "rb") as f:
        artifacts["vectorizer"] = pickle.load(f)

    # Load label encoders
    with open("artifacts/domain_label_encoder.pkl", "rb") as f:
        artifacts["domain_encoder"] = pickle.load(f)
    with open("artifacts/sub_domain_label_encoder.pkl", "rb") as f:
        artifacts["sub_domain_encoder"] = pickle.load(f)

    # Load Classical Models for Domain
    classical_models_domain = {}
    for model_name in ["Logistic", "SVM", "RandomForest", "XGBoost", "NaiveBayes", "KNN"]:
        with open(f"models/domain_{model_name}.pkl", "rb") as f:
            classical_models_domain[model_name] = pickle.load(f)
    artifacts["classical_models_domain"] = classical_models_domain

    # Load Classical Models for Sub-Domain
    classical_models_sub_domain = {}
    for model_name in ["Logistic", "SVM", "RandomForest", "XGBoost", "NaiveBayes", "KNN"]:
        with open(f"models/sub_domain_{model_name}.pkl", "rb") as f:
            classical_models_sub_domain[model_name] = pickle.load(f)
    artifacts["classical_models_sub_domain"] = classical_models_sub_domain

    # Load CatBoost Models
    with open("models/domain_CatBoost.pkl", "rb") as f:
        artifacts["catboost_domain"] = pickle.load(f)
    with open("models/sub_domain_CatBoost.pkl", "rb") as f:
        artifacts["catboost_sub_domain"] = pickle.load(f)

    # Load Tokenizer and max_length for LSTM models
    with open("artifacts/tokenizer.pkl", "rb") as f:
        artifacts["tokenizer"] = pickle.load(f)
    artifacts["max_length"] = 100

    # Load LSTM models
    artifacts["lstm_domain"] = load_model("models/lstm_domain.h5")
    artifacts["lstm_sub_domain"] = load_model("models/lstm_sub_domain.h5")

    # Load Performance Metrics
    with open("artifacts/model_metrics.pkl", "rb") as f:
        artifacts["model_metrics"] = pickle.load(f)

    return artifacts

artifacts = load_artifacts()

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title("🚀 Domain & Sub-Domain Prediction and Model Comparison")

# Create two tabs: Prediction & Performance Comparison
tabs = st.tabs(["Prediction", "Performance Comparison"])

# ----- TAB 1: PREDICTION INTERFACE -----
with tabs[0]:
    st.header("Enter Text for Prediction")
    
    text_input = st.text_area("Enter text:", height=150, placeholder="Type your text here...")
    
    if st.button("Predict"):
        processed_text = preprocess_text(text_input)
        vectorizer = artifacts["vectorizer"]
        vect_text = vectorizer.transform([processed_text])

        domain_encoder = artifacts["domain_encoder"]
        sub_domain_encoder = artifacts["sub_domain_encoder"]

        # --- CatBoost Predictions ---
        catboost_domain_pred = domain_encoder.inverse_transform(
            artifacts["catboost_domain"].predict(vect_text)
        )[0]
        catboost_sub_domain_pred = sub_domain_encoder.inverse_transform(
            artifacts["catboost_sub_domain"].predict(vect_text)
        )[0]

        # Display predictions
        with st.expander("View Predictions", expanded=True):
            st.subheader("CatBoost Predictions")
            st.write(f"**Domain Prediction:** {catboost_domain_pred}")
            st.write(f"**Sub-Domain Prediction:** {catboost_sub_domain_pred}")

print("CatBoost models integrated successfully in deployment.")
