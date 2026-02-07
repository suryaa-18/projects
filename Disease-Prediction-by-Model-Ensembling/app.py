import streamlit as st
import pandas as pd
from src.infer import ensemble_predict

df = pd.read_csv("data/Training.csv")
SYMPTOMS = df.columns[:-1].tolist()

st.set_page_config(page_title="Disease Prediction System")

st.title("🩺 Disease Prediction System")
st.caption("Educational clinical decision-support system")

st.subheader("Select Symptoms")

selected = st.multiselect(
    "Choose symptoms (max 5)",
    SYMPTOMS,
    max_selections=5
)

if st.button("Predict"):
    if len(selected) == 0:
        st.error("Select at least one symptom")
    else:
        # Create binary symptom vector
        vector = [1 if s in selected else 0 for s in SYMPTOMS]

        disease, confidence, top_k = ensemble_predict(vector)

        st.success(f"🩺 Predicted Disease: {disease}")
        st.info(f"Confidence: {confidence * 100:.2f}%")

        if confidence < 0.6:
            st.warning("Low confidence prediction. Please consult a medical professional.")

        st.subheader("Top Predictions")
        for d, p in top_k:
            st.write(f"- {d}: {p * 100:.2f}%")


st.markdown("---")
