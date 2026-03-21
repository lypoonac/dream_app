import os
import re
import random
import html
import numpy as np
import pandas as pd
import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

st.set_page_config(
    page_title="AXA AI Dream Analyzer",
    page_icon="🌙",
    layout="wide",
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGO_PATH = os.path.join(BASE_DIR, "axa_logo.png")

MODEL1_PATH = os.path.join(DATA_DIR, "model1_train.csv")
MODEL2_PATH = os.path.join(DATA_DIR, "model2_train.csv")
SYMBOL_KB_PATH = os.path.join(DATA_DIR, "symbol_kb.csv")

MODEL1_HF_NAME = "peterjerry111/dream-stress-classifier"
GEN_MODEL_NAME = "google/flan-t5-base"

ID2LABEL = {0: "low", 1: "medium", 2: "high"}

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef3fb 100%);
        }
        .brand-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 1.2rem;
            border-left: 6px solid #00008F;
            box-shadow: 0 8px 24px rgba(0, 20, 80, 0.08);
            margin-bottom: 1rem;
        }
        .result-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            border: 1px solid #dbe4f3;
            box-shadow: 0 4px 16px rgba(0, 20, 80, 0.05);
            margin-bottom: 1rem;
        }
        .main-title {
            color: #00008F;
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .sub-title {
            color: #44526b;
            font-size: 1rem;
        }
        .highlight-text {
            background-color: #ffeb3b;
            padding: 4px 8px;
            border-radius: 6px;
            display: inline-block;
            color: #222222;
            font-weight: 600;
        }
        .highlight-block {
            background-color: #ffeb3b;
            padding: 10px 12px;
            border-radius: 8px;
            display: block;
            color: #222222;
            font-weight: 500;
            line-height: 1.7;
        }
        div.stButton > button {
            background-color: #00008F;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 700;
        }
        div.stButton > button:hover {
            background-color: #1f1fb8;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def split_tags(x):
    x = clean_text(x)
    if not x:
        return []
    return [i.strip().lower() for i in x.split(",") if i.strip()]


def tokenize_simple(text):
    return re.findall(r"[a-zA-Z_]+", str(text).lower())


@st.cache_data
def load_data():
    required_files = [MODEL1_PATH, MODEL2_PATH, SYMBOL_KB_PATH]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    df_model1 = pd.read_csv(MODEL1_PATH)
    df_model2 = pd.read_csv(MODEL2_PATH)
    df_symbol_kb = pd.read_csv(SYMBOL_KB_PATH)

    for col in ["dream_text", "stress_label", "emotion_labels", "theme_labels", "symbol_labels"]:
        df_model1[col] = df_model1[col].apply(clean_text)

    df_model1["stress_label"] = df_model1["stress_label"].str.lower().replace({"moderate": "medium"})
    df_model1 = df_model1[df_model1["stress_label"].isin({"low", "medium", "high"})].drop_duplicates().reset_index(drop=True)
    df_model1["emotion_list"] = df_model1["emotion_labels"].apply(split_tags)
    df_model1["theme_list"] = df_model1["theme_labels"].apply(split_tags)
    df_model1["symbol_list"] = df_model1["symbol_labels"].apply(split_tags)

    for col in ["stress_label", "emotion_labels", "dominant_emotion", "recommendation_text"]:
        df_model2[col] = df_model2[col].apply(clean_text)

    df_model2["stress_label"] = df_model2["stress_label"].str.lower().replace({"moderate": "medium"})
    df_model2 = df_model2[df_model2["stress_label"].isin({"low", "medium", "high", "very_high", "severe"})].drop_duplicates().reset_index(drop=True)
    df_model2["emotion_list"] = df_model2["emotion_labels"].apply(split_tags)

    for col in ["symbol_name", "traditional_summary_en", "theme_tags", "emotion_hints", "stress_hint", "source_origin"]:
        df_symbol_kb[col] = df_symbol_kb[col].apply(clean_text)

    df_symbol_kb["symbol_name"] = df_symbol_kb["symbol_name"].str.lower()

    return df_model1, df_model2, df_symbol_kb


@st.cache_resource
def load_models():
    stress_tokenizer = AutoTokenizer.from_pretrained(MODEL1_HF_NAME)
    stress_model = AutoModelForSequenceClassification.from_pretrained(MODEL1_HF_NAME).to(DEVICE)
    stress_model.eval()

    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE)
    gen_model.eval()

    return stress_tokenizer, stress_model, gen_tokenizer, gen_model


def predict_stress(text, tokenizer, model):
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
        pred = torch.argmax(outputs.logits, dim=1).item()

    return ID2LABEL[pred]


def detect_symbols(text, known_symbols):
    tokens = tokenize_simple(text)
    token_set = set(tokens)
    found = []

    for sym in known_symbols:
        if sym in token_set:
            found.append(sym)
        elif "_" in sym:
            parts = sym.split("_")
            if all(part in token_set for part in parts):
                found.append(sym)

    return list(dict.fromkeys(found))


def find_similar_examples(text, df_model1, top_k=5):
    query_tokens = set(tokenize_simple(text))
    scores = []

    for _, row in df_model1.iterrows():
        row_tokens = set(tokenize_simple(row["dream_text"]))
        overlap = len(query_tokens & row_tokens)
        if overlap > 0:
            scores.append((overlap, row))

    scores = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return [row for _, row in scores]


def infer_emotions_themes_symbols(text, df_model1, df_symbol_kb):
    matches = find_similar_examples(text, df_model1, top_k=5)

    emotions, themes, symbols = [], [], []

    for row in matches:
        emotions.extend(row["emotion_list"])
        themes.extend(row["theme_list"])
        symbols.extend(row["symbol_list"])

    emotions = pd.Series(emotions).value_counts().head(5).index.tolist() if emotions else []
    themes = pd.Series(themes).value_counts().head(5).index.tolist() if themes else []
    symbols = pd.Series(symbols).value_counts().head(5).index.tolist() if symbols else []

    direct_symbols = detect_symbols(
        text,
        set(df_symbol_kb["symbol_name"].tolist())
    )

    for s in direct_symbols:
        if s not in symbols:
            symbols.insert(0, s)

    return emotions, themes, symbols


def map_stress_for_model2(stress):
    if stress == "low":
        return ["low"]
    if stress == "medium":
        return ["medium"]
    if stress == "high":
        return ["high", "very_high"]
    return ["medium"]


def retrieve_recommendation(pred_stress, inferred_emotions, df_model2):
    stress_candidates = map_stress_for_model2(pred_stress)
    subset = df_model2[df_model2["stress_label"].isin(stress_candidates)].copy()

    if subset.empty:
        return "Take things one step at a time, reduce pressure, and focus on steady emotional recovery."

    emo_set = set(inferred_emotions)
    scored = []

    for _, row in subset.iterrows():
        overlap = len(emo_set.intersection(set(row["emotion_list"])))
        score = overlap + (1 if row["dominant_emotion"] in emo_set else 0)
        scored.append((score, row))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    return scored[0][1]["recommendation_text"]


def build_recommendation_prompt(dream_text, stress, emotions, themes, symbols):
    emotions_text = ", ".join(emotions[:5]) if emotions else "none clearly inferred"
    themes_text = ", ".join([t.replace("_", " ") for t in themes[:5]]) if themes else "none clearly inferred"
    symbols_text = ", ".join([s.replace("_", " ") for s in symbols[:5]]) if symbols else "none clearly inferred"

    return f"""
Give a short supportive recommendation based on this dream analysis.

Dream: {dream_text}
Stress level: {stress}
Emotions: {emotions_text}
Themes: {themes_text}
Symbols: {symbols_text}

Write only the recommendation in 2 to 3 sentences.
Be calm, practical, and supportive.
Do not repeat the prompt.
""".strip()


def generate_text(prompt, tokenizer, model, max_new_tokens=100):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def is_bad_recommendation(text):
    if not text:
        return True

    low = text.lower().strip()

    bad_prefixes = [
        "you are",
        "dream:",
        "stress level:",
        "emotions:",
        "themes:",
        "symbols:",
        "give a short supportive recommendation",
        "write only the recommendation",
    ]

    if any(low.startswith(prefix) for prefix in bad_prefixes):
        return True

    if len(text.split()) < 6:
        return True

    return False


def analyze_dream(
    dream_text,
    stress_tokenizer,
    stress_model,
    gen_tokenizer,
    gen_model,
    df_model1,
    df_model2,
    df_symbol_kb,
):
    stress = predict_stress(dream_text, stress_tokenizer, stress_model)
    emotions, themes, symbols = infer_emotions_themes_symbols(
        dream_text, df_model1, df_symbol_kb
    )

    prompt = build_recommendation_prompt(
        dream_text=dream_text,
        stress=stress,
        emotions=emotions,
        themes=themes,
        symbols=symbols,
    )

    generated_recommendation = generate_text(
        prompt,
        gen_tokenizer,
        gen_model,
        max_new_tokens=100,
    )

    fallback_recommendation = retrieve_recommendation(stress, emotions, df_model2)

    if is_bad_recommendation(generated_recommendation):
        final_recommendation = fallback_recommendation
    else:
        final_recommendation = generated_recommendation

    return {
        "stress_level": stress,
        "emotions": emotions,
        "themes": themes,
        "symbols": symbols,
        "recommendation": final_recommendation,
    }


try:
    df_model1, df_model2, df_symbol_kb = load_data()
    stress_tokenizer, stress_model, gen_tokenizer, gen_model = load_models()
except Exception as e:
    st.error("Failed to load files or models.")
    st.exception(e)
    st.stop()


st.markdown(
    """
    <div class="brand-card">
        <div class="main-title">AXA AI Dream Analyzer</div>
        <div class="sub-title">
            Dream-based stress detection with supportive recommendation output.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

dream_text = st.text_area(
    "Enter your dream",
    height=180,
    placeholder="Describe your dream here..."
)

if st.button("Analyze Dream"):
    if not dream_text.strip():
        st.warning("Please enter a dream.")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_dream(
                dream_text.strip(),
                stress_tokenizer,
                stress_model,
                gen_tokenizer,
                gen_model,
                df_model1,
                df_model2,
                df_symbol_kb,
            )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Stress Level")
            st.markdown(
                f"<div class='highlight-text'>{html.escape(result['stress_level'].upper())}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Emotions")
            emotions_text = ", ".join(result["emotions"]) if result["emotions"] else "None"
            st.markdown(
                f"<div class='highlight-text'>{html.escape(emotions_text)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Themes")
        st.write(", ".join(result["themes"]) if result["themes"] else "None")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Symbols")
        st.write(", ".join(result["symbols"]) if result["symbols"] else "None")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Recommendation")
        st.markdown(
            f"<div class='highlight-block'>{html.escape(result['recommendation'])}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
