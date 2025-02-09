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

def plot_lstm_history(history, title="LSTM Training History"):
    """Plot training and validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Plot
    axes[0].plot(history.get("loss", []), marker="o", label="Train Loss")
    axes[0].plot(history.get("val_loss", []), marker="o", label="Val Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy Plot
    axes[1].plot(history.get("accuracy", []), marker="o", label="Train Accuracy")
    axes[1].plot(history.get("val_accuracy", []), marker="o", label="Val Accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    return fig

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

        # --- Classical Models Prediction ---
        vectorizer = artifacts["vectorizer"]
        vect_text = vectorizer.transform([processed_text])

        domain_encoder = artifacts["domain_encoder"]
        sub_domain_encoder = artifacts["sub_domain_encoder"]
        classical_models_domain = artifacts["classical_models_domain"]
        classical_models_sub_domain = artifacts["classical_models_sub_domain"]

        classical_domain_preds = {
            name: domain_encoder.inverse_transform(model.predict(vect_text))[0]
            for name, model in classical_models_domain.items()
        }
        classical_sub_domain_preds = {
            name: sub_domain_encoder.inverse_transform(model.predict(vect_text))[0]
            for name, model in classical_models_sub_domain.items()
        }

        # --- LSTM Models Prediction ---
        tokenizer = artifacts["tokenizer"]
        max_length = artifacts["max_length"]
        seq = tokenizer.texts_to_sequences([processed_text])
        padded_seq = pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")

        lstm_domain_pred = domain_encoder.inverse_transform(
            [np.argmax(artifacts["lstm_domain"].predict(padded_seq)[0])]
        )[0]
        lstm_sub_domain_pred = sub_domain_encoder.inverse_transform(
            [np.argmax(artifacts["lstm_sub_domain"].predict(padded_seq)[0])]
        )[0]

        # Display predictions
        with st.expander("View Predictions", expanded=True):
            st.subheader("Classical Models Predictions")
            st.write(f"**Domain Prediction:**")
            st.json(classical_domain_preds)
            st.write(f"**Sub-Domain Prediction:**")
            st.json(classical_sub_domain_preds)

            st.subheader("LSTM Models Predictions")
            st.write(f"**Domain Prediction:** {lstm_domain_pred}")
            st.write(f"**Sub-Domain Prediction:** {lstm_sub_domain_pred}")

# ----- TAB 2: PERFORMANCE COMPARISON -----
with tabs[1]:
    st.header("Performance Metrics & Comparison Graphs")
    
    metrics = artifacts["model_metrics"]

    for model_type in ["classical_domain", "classical_sub_domain"]:
        if metrics.get(model_type):
            df = pd.DataFrame({
                "Model": list(metrics[model_type].keys()),
                "Accuracy": [metrics[model_type][m]["accuracy"] for m in metrics[model_type]],
                "F1 Score": [metrics[model_type][m]["f1"] for m in metrics[model_type]]
            }).set_index("Model")
            st.subheader(f"{model_type.replace('_', ' ').title()} Performance")
            st.table(df)
            st.bar_chart(df)

    for lstm_type in ["lstm_domain", "lstm_sub_domain"]:
        if metrics.get(lstm_type):
            st.subheader(f"{lstm_type.replace('_', ' ').title()} Performance")
            st.write(f"Test Accuracy: {metrics[lstm_type].get('test_accuracy', 0):.4f}")
            fig = plot_lstm_history(metrics[lstm_type].get("train_history", {}), title=lstm_type.replace('_', ' ').title())
            st.pyplot(fig)
