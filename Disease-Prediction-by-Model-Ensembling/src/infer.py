# ================================
# ENSEMBLE INFERENCE SCRIPT
# ================================

import joblib
import numpy as np
import os

MODEL_DIR = "models"

# -------------------------------
# Load trained models
# -------------------------------
nb = joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl"))
dt = joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))

label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
symptoms = joblib.load(os.path.join(MODEL_DIR, "symptoms.pkl"))


# -------------------------------
# Ensemble prediction function
# -------------------------------
def ensemble_predict(symptom_vector):
    """
    symptom_vector: list or array of shape (n_symptoms,)
    returns:
        predicted_disease (str)
        confidence (float)
        top_k (list of tuples)
    """

    x = np.array(symptom_vector).reshape(1, -1)

    # Individual model probabilities
    p_nb = nb.predict_proba(x)
    p_dt = dt.predict_proba(x)
    p_rf = rf.predict_proba(x)

    # Soft voting (probability averaging)
    avg_prob = (p_nb + p_dt + p_rf) / 3

    # Final prediction
    pred_index = np.argmax(avg_prob)
    confidence = avg_prob[0][pred_index]

    predicted_disease = label_encoder.inverse_transform([pred_index])[0]

    # Top-3 predictions
    top_indices = np.argsort(avg_prob[0])[::-1][:3]
    top_k = [
        (label_encoder.inverse_transform([i])[0], avg_prob[0][i])
        for i in top_indices
    ]

    return predicted_disease, confidence, top_k
