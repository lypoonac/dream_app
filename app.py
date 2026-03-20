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
    page_title="AXA AI Dream Analyzer: Early Stress Detection",
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
MODEL2_HF_NAME = "google/flan-t5-base"

ID2LABEL = {0: "low", 1: "medium", 2: "high"}

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef3fb 100%);
        }
        .main-title {
            color: #00008F;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.3rem;
        }
        .sub-title {
            color: #44526b;
            font-size: 1rem;
            margin-bottom: 0.8rem;
        }
        .brand-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            border-left: 6px solid #00008F;
            box-shadow: 0 8px 24px rgba(0, 20, 80, 0.08);
            margin-bottom: 1rem;
        }
        .guide-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            border: 1px solid #dbe4f3;
            box-shadow: 0 4px 16px rgba(0, 20, 80, 0.05);
            margin-bottom: 1rem;
        }
        .result-card {
            background: #fff9c4;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            border: 1px solid #f4e287;
            box-shadow: 0 4px 16px rgba(120, 100, 0, 0.08);
            margin-bottom: 1rem;
        }
        .section-title {
            color: #00008F;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .small-muted {
            color: #5f6c86;
            font-size: 0.95rem;
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
        textarea {
            border-radius: 12px !important;
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
    missing = []
    for p in [MODEL1_PATH, MODEL2_PATH, SYMBOL_KB_PATH]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    df_model1 = pd.read_csv(MODEL1_PATH)
    df_model2 = pd.read_csv(MODEL2_PATH)
    df_symbol_kb = pd.read_csv(SYMBOL_KB_PATH)

    for col in ["dream_text", "stress_label", "emotion_labels", "theme_labels", "symbol_labels"]:
        if col not in df_model1.columns:
            raise ValueError(f"Column '{col}' missing from model1_train.csv")
        df_model1[col] = df_model1[col].apply(clean_text)

    df_model1["stress_label"] = df_model1["stress_label"].str.lower().replace({"moderate": "medium"})
    df_model1 = df_model1[
        df_model1["stress_label"].isin({"low", "medium", "high"})
    ].drop_duplicates().reset_index(drop=True)
    df_model1["emotion_list"] = df_model1["emotion_labels"].apply(split_tags)

    for col in ["stress_label", "emotion_labels", "dominant_emotion", "recommendation_text"]:
        if col not in df_model2.columns:
            raise ValueError(f"Column '{col}' missing from model2_train.csv")
        df_model2[col] = df_model2[col].apply(clean_text)

    df_model2["stress_label"] = df_model2["stress_label"].str.lower().replace({"moderate": "medium"})
    df_model2 = df_model2.drop_duplicates().reset_index(drop=True)

    for col in ["symbol_name", "traditional_summary_en", "theme_tags", "emotion_hints", "stress_hint", "source_origin"]:
        if col not in df_symbol_kb.columns:
            raise ValueError(f"Column '{col}' missing from symbol_kb.csv")
        df_symbol_kb[col] = df_symbol_kb[col].apply(clean_text)

    df_symbol_kb["symbol_name"] = df_symbol_kb["symbol_name"].str.lower()
    df_symbol_kb["emotion_list"] = df_symbol_kb["emotion_hints"].apply(
        lambda x: [i.strip().lower() for i in clean_text(x).split(",") if i.strip()]
    )

    return df_model1, df_model2, df_symbol_kb


@st.cache_resource
def load_models():
    stress_tokenizer = AutoTokenizer.from_pretrained(MODEL1_HF_NAME)
    stress_model = AutoModelForSequenceClassification.from_pretrained(MODEL1_HF_NAME).to(DEVICE)
    stress_model.eval()

    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL2_HF_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL2_HF_NAME).to(DEVICE)
    gen_model.eval()

    return stress_tokenizer, stress_model, gen_tokenizer, gen_model


def predict_stress(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = int(torch.argmax(outputs.logits, dim=1).item())

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


def infer_emotions(text, df_model1, df_symbol_kb):
    matches = find_similar_examples(text, df_model1, top_k=5)

    emotions = []
    for row in matches:
        emotions.extend(row["emotion_list"])

    emotions = pd.Series(emotions).value_counts().head(5).index.tolist() if emotions else []

    direct_symbols = detect_symbols(text, set(df_symbol_kb["symbol_name"].tolist()))
    if direct_symbols:
        symbol_emotions = []
        matched_symbol_rows = df_symbol_kb[df_symbol_kb["symbol_name"].isin(direct_symbols)]
        for _, row in matched_symbol_rows.iterrows():
            symbol_emotions.extend(row["emotion_list"])

        if symbol_emotions:
            combined = emotions + symbol_emotions
            emotions = pd.Series(combined).value_counts().head(5).index.tolist()

    return emotions


def build_dream_interpretation_prompt(dream_text, stress, emotions):
    emotions_text = ", ".join(emotions[:5]) if emotions else "unclear emotions"

    prompt = f"""
You are writing a short dream interpretation.

Task:
Write a natural dream interpretation based on the dream narrative, stress level, and emotions.

Rules:
- Write exactly 2 sentences.
- Sound like a human dream interpreter.
- Focus on symbolism, inner feelings, and possible emotional meaning.
- Be thoughtful, reflective, and easy to understand.
- Do not give advice.
- Do not mention stress classification, model, AI, analysis, prompt, or dataset.
- Do not use bullet points.
- Do not sound robotic or repetitive.
- Use tentative interpretation words such as "may", "could", "might", or "suggests".
- Make it sound like dream interpretation, not a recommendation.

Dream narrative: {dream_text}
Stress level: {stress}
Emotions: {emotions_text}

Dream interpretation:
""".strip()

    return prompt


def postprocess_interpretation(text):
    text = text.strip()

    bad_prefixes = [
        "dream interpretation:",
        "interpretation:",
        "response:",
        "answer:",
    ]
    lower_text = text.lower()
    for prefix in bad_prefixes:
        if lower_text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > 0:
        text = text[0].upper() + text[1:]

    return text


def generate_text(prompt, tokenizer, model, max_new_tokens=90):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
            no_repeat_ngram_size=3,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return text


def analyze_dream(
    dream_text,
    stress_tokenizer,
    stress_model,
    gen_tokenizer,
    gen_model,
    df_model1,
    df_symbol_kb,
):
    stress = predict_stress(dream_text, stress_tokenizer, stress_model)
    emotions = infer_emotions(dream_text, df_model1, df_symbol_kb)

    insight_prompt = build_dream_interpretation_prompt(
        dream_text=dream_text,
        stress=stress,
        emotions=emotions,
    )

    dream_interpretation = generate_text(
        insight_prompt,
        gen_tokenizer,
        gen_model,
        max_new_tokens=90,
    )

    dream_interpretation = postprocess_interpretation(dream_interpretation)

    return {
        "dream_text": dream_text,
        "stress_level": stress,
        "emotions": emotions,
        "dream_interpretation": dream_interpretation,
    }


try:
    df_model1, df_model2, df_symbol_kb = load_data()
    stress_tokenizer, stress_model, gen_tokenizer, gen_model = load_models()
except Exception as e:
    st.error("Failed to load required files or models.")
    st.exception(e)
    st.stop()


if os.path.exists(LOGO_PATH):
    header_col1, header_col2 = st.columns([1, 4])
    with header_col1:
        st.image(LOGO_PATH, width=140)
    with header_col2:
        st.markdown(
            """
            <div class="brand-card">
                <div class="main-title">AXA AI Dream Analyzer: Early Stress Detection</div>
                <div class="sub-title">
                    A prototype wellness support tool that analyzes dream narratives to surface
                    possible stress signals, emotional patterns, and dream interpretation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div class="brand-card">
            <div class="main-title">AXA AI Dream Analyzer: Early Stress Detection</div>
            <div class="sub-title">
                A prototype wellness support tool that analyzes dream narratives to surface
                possible stress signals, emotional patterns, and dream interpretation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="guide-card">
        <div class="section-title">Simple User Guide</div>
        <div class="small-muted">
            1. Type your dream into the text box below.<br>
            2. Click <b>Analyze Dream</b> to start the analysis.<br>
            3. Review the predicted stress level and inferred emotions.<br>
            4. Read the generated dream interpretation.<br>
            5. This tool supports reflection and early stress awareness, not diagnosis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

dream_text = st.text_area(
    "Enter your dream",
    value="",
    height=180,
    placeholder="Describe your dream here..."
)

if st.button("Analyze Dream"):
    if not dream_text.strip():
        st.warning("Please enter a dream before analysis.")
    else:
        with st.spinner("Analyzing your dream..."):
            result = analyze_dream(
                dream_text=dream_text.strip(),
                stress_tokenizer=stress_tokenizer,
                stress_model=stress_model,
                gen_tokenizer=gen_tokenizer,
                gen_model=gen_model,
                df_model1=df_model1,
                df_symbol_kb=df_symbol_kb,
            )

        st.success("Analysis completed.")

        stress_text = html.escape(result["stress_level"].upper())
        emotions_text = html.escape(", ".join(result["emotions"]) if result["emotions"] else "None")
        interpretation_text = html.escape(result["dream_interpretation"])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Stress Level")
            st.markdown(
                f"<div class='highlight-text'>{stress_text}</div>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Emotions")
            st.markdown(
                f"<div class='highlight-text'>{emotions_text}</div>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Dream Interpretation")
        st.markdown(
            f"<div class='highlight-block'>{interpretation_text}</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
