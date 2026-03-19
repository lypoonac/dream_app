import os
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# CONFIG
# =========================
DATA_PATH = "data/model1_train.csv"

TEXT_COL = "dream_text"
LABEL_COL = "stress_label"

MODEL_REPO = "peterjerry111/dream-stress-classifier"

RANDOM_STATE = 42
TEST_SIZE = 0.2

ALLOWED_LABELS = {"low", "medium", "high"}
LABEL_NORMALIZATION = {
    "moderate": "medium"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# HELPERS
# =========================
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def normalize_label(x):
    x = clean_text(x).lower()
    return LABEL_NORMALIZATION.get(x, x)

def load_dataset():
    df = pd.read_csv(DATA_PATH)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing text column: {TEXT_COL}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    df[LABEL_COL] = df[LABEL_COL].apply(normalize_label)

    df = df[df[LABEL_COL].isin(ALLOWED_LABELS)].copy()
    df = df[df[TEXT_COL] != ""].copy()
    df = df.drop_duplicates().reset_index(drop=True)

    return df

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def predict_label(text, tokenizer, model):
    id2label = model.config.id2label

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = int(torch.argmax(outputs.logits, dim=1).item())

    pred_label = id2label[pred_id]

    if isinstance(pred_label, str):
        pred_label = pred_label.lower().strip()

    pred_label = LABEL_NORMALIZATION.get(pred_label, pred_label)
    return pred_label

def main():
    df = load_dataset()

    X = df[TEXT_COL]
    y = df[LABEL_COL]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    tokenizer, model = load_model()

    preds = []
    for text in X_test:
        preds.append(predict_label(text, tokenizer, model))

    accuracy = accuracy_score(y_test, preds)

    results = {
        "final_pipeline": "Hugging Face classifier + retrieval/symbol pipeline",
        "platform": "Streamlit Cloud / deployment-equivalent environment",
        "sample_size": len(X_test),
        "accuracy": accuracy
    }

    os.makedirs("results", exist_ok=True)

    with open("results/final_app_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL APPLICATION RESULTS ===")
    print(json.dumps(results, indent=2))
    print("\nSaved: results/final_app_results.json")

if __name__ == "__main__":
    main()
