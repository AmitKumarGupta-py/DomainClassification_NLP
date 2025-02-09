import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Classical Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Deep Learning Modules
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set seaborn style for attractive plots
sns.set(style="whitegrid")

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_text(text):
    """Clean text by lowercasing and removing non-alphanumeric characters."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def build_lstm_model(max_features, max_length, num_classes):
    """Construct and compile an LSTM model."""
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_length))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# -----------------------------
# Main Training Pipeline
# -----------------------------
def main():
    # Streamlit UI
    st.title("Model Training Progress")
    st.sidebar.title("Progress Overview")
    
    # 1. Load Dataset
    st.sidebar.write("Step 1: Loading dataset...")
    df = pd.read_csv("final_dataset.csv")
    st.sidebar.write(f"Dataset loaded: {len(df)} records")
    
    # 2. Preprocess Text
    st.sidebar.write("Step 2: Preprocessing text...")
    df["cleaned_text"] = df["description"].apply(preprocess_text)
    
    # 3. Feature Extraction for Classical Models (TF-IDF)
    st.sidebar.write("Step 3: Extracting features using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(df["cleaned_text"])
    with open("artifacts/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    st.sidebar.write("TF-IDF vectorizer saved.")
    
    # 4. Label Encoding for Domain and Sub-Domain
    st.sidebar.write("Step 4: Label encoding...")
    domain_encoder = LabelEncoder()
    sub_domain_encoder = LabelEncoder()
    domain_encoded = domain_encoder.fit_transform(df["domain"])
    sub_domain_encoded = sub_domain_encoder.fit_transform(df["sub_domain"])
    
    with open("artifacts/domain_label_encoder.pkl", "wb") as f:
        pickle.dump(domain_encoder, f)
    with open("artifacts/sub_domain_label_encoder.pkl", "wb") as f:
        pickle.dump(sub_domain_encoder, f)
    st.sidebar.write("Label encoders saved.")
    
    # 5. Split Data (same split for both tasks)
    st.sidebar.write("Step 5: Splitting data...")
    X_train, X_test, y_train_domain, y_test_domain, y_train_sub_domain, y_test_sub_domain = train_test_split(
        X_tfidf, domain_encoded, sub_domain_encoded, test_size=0.2, random_state=42
    )
    
    # Dictionary to store performance metrics
    model_metrics = {
        "classical_domain": {},
        "classical_sub_domain": {},
        "lstm_domain": {},
        "lstm_sub_domain": {}
    }
    
    # 6. Train and Save Classical Models for Domain Prediction
    classical_models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False),
        "NaiveBayes": MultinomialNB(),
        "KNN": KNeighborsClassifier()
    }
    
    st.sidebar.write("Step 6: Training classical models for domain prediction...")
    with st.spinner("Training classical models... This may take some time..."):
        for name, model in classical_models.items():
            st.sidebar.write(f"Training {name} for DOMAIN...")
            model.fit(X_train, y_train_domain)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test_domain, y_pred)
            f1 = f1_score(y_test_domain, y_pred, average="weighted")
            st.sidebar.write(f"[DOMAIN] {name}: Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
            model_metrics["classical_domain"][name] = {"accuracy": acc, "f1": f1}
            with open(f"models/domain_{name}.pkl", "wb") as f:
                pickle.dump(model, f)
    
    st.sidebar.write("Step 7: Training classical models for sub-domain prediction...")
    with st.spinner("Training classical models for sub-domain prediction..."):
        for name, model in classical_models.items():
            st.sidebar.write(f"Training {name} for SUB-DOMAIN...")
            model.fit(X_train, y_train_sub_domain)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test_sub_domain, y_pred)
            f1 = f1_score(y_test_sub_domain, y_pred, average="weighted")
            st.sidebar.write(f"[SUB-DOMAIN] {name}: Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
            model_metrics["classical_sub_domain"][name] = {"accuracy": acc, "f1": f1}
            with open(f"models/sub_domain_{name}.pkl", "wb") as f:
                pickle.dump(model, f)
    
    # 7. Prepare Data for LSTM Models
    st.sidebar.write("Step 8: Preparing data for LSTM models...")
    max_features = 10000
    max_length = 100
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["cleaned_text"])
    sequences = tokenizer.texts_to_sequences(df["cleaned_text"])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    with open("artifacts/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    st.sidebar.write("Tokenizer saved.")
    
    # 8. Train and Save LSTM Model for Domain Prediction
    st.sidebar.write("Step 9: Training LSTM model for domain prediction...")
    with st.spinner("Training LSTM model for domain prediction..."):
        X_lstm_train, X_lstm_test, y_lstm_train_domain, y_lstm_test_domain = train_test_split(
            padded_sequences, domain_encoded, test_size=0.2, random_state=42
        )
        num_classes_domain = len(np.unique(domain_encoded))
        lstm_domain = build_lstm_model(max_features, max_length, num_classes_domain)
        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        history_domain = lstm_domain.fit(
            X_lstm_train, y_lstm_train_domain,
            epochs=10, batch_size=32,
            validation_split=0.2, callbacks=[early_stop],
            verbose=1
        )
        loss, acc = lstm_domain.evaluate(X_lstm_test, y_lstm_test_domain, verbose=0)
        st.sidebar.write(f"LSTM Domain Model: Test Accuracy = {acc:.4f}")
        model_metrics["lstm_domain"]["test_accuracy"] = acc
        model_metrics["lstm_domain"]["train_history"] = history_domain.history
        lstm_domain.save("models/lstm_domain.h5")
    
    # 9. Train and Save LSTM Model for Sub-Domain Prediction
    st.sidebar.write("Step 10: Training LSTM model for sub-domain prediction...")
    with st.spinner("Training LSTM model for sub-domain prediction..."):
        X_lstm_train_sub, X_lstm_test_sub, y_lstm_train_sub, y_lstm_test_sub = train_test_split(
            padded_sequences, sub_domain_encoded, test_size=0.2, random_state=42
        )
        num_classes_sub_domain = len(np.unique(sub_domain_encoded))
        lstm_sub_domain = build_lstm_model(max_features, max_length, num_classes_sub_domain)
        history_sub_domain = lstm_sub_domain.fit(
            X_lstm_train_sub, y_lstm_train_sub,
            epochs=10, batch_size=32,
            validation_split=0.2, callbacks=[early_stop],
            verbose=1
        )
        loss, acc = lstm_sub_domain.evaluate(X_lstm_test_sub, y_lstm_test_sub, verbose=0)
        st.sidebar.write(f"LSTM Sub-Domain Model: Test Accuracy = {acc:.4f}")
        model_metrics["lstm_sub_domain"]["test_accuracy"] = acc
        model_metrics["lstm_sub_domain"]["train_history"] = history_sub_domain.history
        lstm_sub_domain.save("models/lstm_sub_domain.h5")
    
    # 10. Save performance metrics for later use
    with open("artifacts/model_metrics.pkl", "wb") as f:
        pickle.dump(model_metrics, f)
    st.sidebar.write("\nAll models, artifacts, and performance metrics have been saved successfully.")
    
    # 11. Generate and Save Comparison Charts for Classical Models (Domain & Sub-Domain)
    for task, metric_key in zip(["Domain", "Sub-Domain"], ["classical_domain", "classical_sub_domain"]):
        metrics_data = model_metrics[metric_key]
        models_list = list(metrics_data.keys())
        accuracies = [metrics_data[m]["accuracy"] for m in models_list]
        f1_scores = [metrics_data[m]["f1"] for m in models_list]
        df_metrics = pd.DataFrame({
            "Model": models_list,
            "Accuracy": accuracies,
            "F1 Score": f1_scores
        }).set_index("Model")
        plt.figure(figsize=(10, 5))
        df_metrics.plot(kind="bar", rot=0, color=["#4c72b0", "#55a868"])
        plt.title(f"Classical Models Performance for {task} Prediction")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"artifacts/classical_{task.lower().replace('-', '_')}_performance.png")
        plt.close()
        st.sidebar.write(f"Saved performance chart for {task} models.")

if __name__ == "__main__":
    main()
