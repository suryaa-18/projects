# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SYMPTOMS = None

def load_and_preprocess(csv_path):
    global SYMPTOMS

    df = pd.read_csv(csv_path)
    SYMPTOMS = df.columns[:-1].tolist()

    X = df[SYMPTOMS]
    y = df["prognosis"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X, y_enc, le, SYMPTOMS
