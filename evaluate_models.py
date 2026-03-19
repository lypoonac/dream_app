import os
import time
import json
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_PATH = "data/model1_train.csv"
TEXT_COL = "dream_text"
LABEL_COL = "stress_label"
MODEL_REPO = "your-username/dream-stress-classifier"
RANDOM_STATE = 42
TEST_SIZE = 0.2

ALLOWED_LABELS = {"low", "medium", "high"}
LABEL_NORMALIZATION = {"moderate": "medium"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def evaluate_sklearn_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)

    start = time.time()
    preds = model.predict(X_test)
    end = time.time()

    accuracy = accuracy_score(y_test, preds)
    total_runtime = end - start
    avg_runtime = total_runtime / len(X_test)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "total_runtime_sec": total_runtime,
        "avg_runtime_per_sample_sec": avg_runtime
    }


def load_hf_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_hf(text, tokenizer, model):
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


def evaluate_hf_model(X_test, y_test, model_name):
    tokenizer, model = load_hf_model()
    preds = []

    start = time.time()
    for text in X_test:
        preds.append(predict_hf(text, tokenizer, model))
    end = time.time()

    accuracy = accuracy_score(y_test, preds)
    total_runtime = end - start
    avg_runtime = total_runtime / len(X_test)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "total_runtime_sec": total_runtime,
        "avg_runtime_per_sample_sec": avg_runtime
    }


def run_model_selection():
    df = load_dataset()

    X = df[TEXT_COL]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    results = []

    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    results.append(
        evaluate_sklearn_model(
            lr_pipeline, X_train, X_test, y_train, y_test,
            "TF-IDF + Logistic Regression"
        )
    )

    svm_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LinearSVC())
    ])
    results.append(
        evaluate_sklearn_model(
            svm_pipeline, X_train, X_test, y_train, y_test,
            "TF-IDF + Linear SVM"
        )
    )

    results.append(
        evaluate_hf_model(
            list(X_test), list(y_test),
            "Hugging Face Transformer"
        )
    )

    results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/model_selection_results.csv", index=False)

    metadata = {
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": RANDOM_STATE,
        "test_size_ratio": TEST_SIZE
    }

    with open("results/model_selection_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return results_df, metadata


if __name__ == "__main__":
    df_results, meta = run_model_selection()
    print(df_results)
    print(meta)
