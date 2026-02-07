# ================================
# TRAINING SCRIPT
# ================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "data/Training.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42


# -------------------------------
# 1. LOAD DATA
# -------------------------------
print("📥 Loading dataset...")

df = pd.read_csv(DATA_PATH)

assert "prognosis" in df.columns, "❌ 'prognosis' column missing"

SYMPTOMS = df.columns[:-1].tolist()
X = df[SYMPTOMS]
y = df["prognosis"]

print(f"✔ Samples: {df.shape[0]}")
print(f"✔ Symptoms: {len(SYMPTOMS)}")
print(f"✔ Diseases: {y.nunique()}")


# -------------------------------
# 2. LABEL ENCODING
# -------------------------------
print("\n🔄 Encoding disease labels...")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("✔ Classes:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"  {i}: {cls}")


# -------------------------------
# 3. LOAD EXTERNAL TEST DATA
# -------------------------------
print("\n📥 Loading Testing dataset...")

df_test = pd.read_csv("data/Testing.csv")

assert "prognosis" in df_test.columns, "❌ 'prognosis' missing in Testing.csv"

# Ensure column consistency
assert list(df_test.columns[:-1]) == SYMPTOMS, \
    "❌ Symptom columns in Testing.csv do not match Training.csv"

X_test = df_test[SYMPTOMS]
y_test = label_encoder.transform(df_test["prognosis"])

print(f"✔ Test samples: {df_test.shape[0]}")


# -------------------------------
# 4. INITIALIZE MODELS
# -------------------------------
print("\n🧠 Initializing models...")

models = {
    "naive_bayes": GaussianNB(),
    "decision_tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=None
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# -------------------------------
# 5. TRAIN & VALIDATE
# -------------------------------
print("\n🚀 Training on Training.csv and validating on Testing.csv...\n")

os.makedirs(MODEL_DIR, exist_ok=True)

for name, model in models.items():
    print(f"▶ Training {name}...")

    # Train on FULL training data
    model.fit(X, y_encoded)

    # Evaluate on Testing.csv
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✔ Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    # Save trained model
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Saved to {model_path}\n")



# -------------------------------
# 6. SAVE LABEL ENCODER & METADATA
# -------------------------------
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
joblib.dump(SYMPTOMS, os.path.join(MODEL_DIR, "symptoms.pkl"))

print("✅ Training complete")
print("📦 Models and encoders saved in /models")
