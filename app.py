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
SYMBOL_KB_PATH = os.path.join(DATA_DIR, "symbol_kb.csv")
DREAM_INTERPRETATIONS_PATH = os.path.join(DATA_DIR, "dreams_interpretations.csv")

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
            background: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            border: 1px solid #dbe4f3;
            box-shadow: 0 4px 16px rgba(0, 20, 80, 0.05);
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


def normalize_symbol(text):
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s/&'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data
def load_data():
    missing = []
    for p in [MODEL1_PATH, SYMBOL_KB_PATH, DREAM_INTERPRETATIONS_PATH]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    df_model1 = pd.read_csv(MODEL1_PATH)
    df_symbol_kb = pd.read_csv(SYMBOL_KB_PATH)
    df_interp = pd.read_csv(DREAM_INTERPRETATIONS_PATH)

    for col in ["dream_text", "stress_label", "emotion_labels", "theme_labels", "symbol_labels"]:
        if col not in df_model1.columns:
            raise ValueError(f"Column '{col}' missing from model1_train.csv")
        df_model1[col] = df_model1[col].apply(clean_text)

    df_model1["stress_label"] = df_model1["stress_label"].str.lower().replace({"moderate": "medium"})
    df_model1 = df_model1[
        df_model1["stress_label"].isin({"low", "medium", "high"})
    ].drop_duplicates().reset_index(drop=True)
    df_model1["emotion_list"] = df_model1["emotion_labels"].apply(split_tags)

    for col in ["symbol_name", "traditional_summary_en", "theme_tags", "emotion_hints", "stress_hint", "source_origin"]:
        if col not in df_symbol_kb.columns:
            raise ValueError(f"Column '{col}' missing from symbol_kb.csv")
        df_symbol_kb[col] = df_symbol_kb[col].apply(clean_text)

    df_symbol_kb["symbol_name"] = df_symbol_kb["symbol_name"].str.lower()
    df_symbol_kb["emotion_list"] = df_symbol_kb["emotion_hints"].apply(
        lambda x: [i.strip().lower() for i in clean_text(x).split(",") if i.strip()]
    )

    interp_symbol_col = None
    interp_text_col = None

    for c in df_interp.columns:
        c_clean = c.strip().lower()
        if c_clean == "dream symbol":
            interp_symbol_col = c
        if c_clean == "interpretation":
            interp_text_col = c

    if interp_symbol_col is None or interp_text_col is None:
        raise ValueError("dreams_interpretations.csv must contain 'Dream Symbol' and 'Interpretation' columns")

    df_interp = df_interp[[interp_symbol_col, interp_text_col]].copy()
    df_interp.columns = ["dream_symbol", "interpretation"]
    df_interp["dream_symbol"] = df_interp["dream_symbol"].apply(clean_text)
    df_interp["interpretation"] = df_interp["interpretation"].apply(clean_text)
    df_interp = df_interp[(df_interp["dream_symbol"] != "") & (df_interp["interpretation"] != "")]
    df_interp["symbol_norm"] = df_interp["dream_symbol"].apply(normalize_symbol)
    df_interp = df_interp.drop_duplicates(subset=["symbol_norm", "interpretation"]).reset_index(drop=True)

    return df_model1, df_symbol_kb, df_interp


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
        sym_clean = sym.lower()
        if sym_clean in token_set:
            found.append(sym_clean)
        elif "_" in sym_clean:
            parts = sym_clean.split("_")
            if all(part in token_set for part in parts):
                found.append(sym_clean)

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


def find_interpretation_matches(dream_text, df_interp, max_matches=5):
    text_lower = f" {normalize_symbol(dream_text)} "
    matches = []

    for _, row in df_interp.iterrows():
        sym = row["symbol_norm"]
        if not sym:
            continue

        pattern = f" {sym} "
        if pattern in text_lower:
            matches.append((len(sym), row["dream_symbol"], row["interpretation"]))

    matches = sorted(matches, key=lambda x: x[0], reverse=True)

    dedup = []
    seen = set()
    for _, symbol, interpretation in matches:
        key = normalize_symbol(symbol)
        if key not in seen:
            seen.add(key)
            dedup.append((symbol, interpretation))
        if len(dedup) >= max_matches:
            break

    return dedup


def build_dataset_grounded_prompt(dream_text, stress, emotions, matched_interpretations):
    emotions_text = ", ".join(emotions[:5]) if emotions else "unclear emotions"

    if matched_interpretations:
        symbol_lines = []
        for symbol, interpretation in matched_interpretations:
            symbol_lines.append(f"- {symbol}: {interpretation}")
        symbol_block = "\n".join(symbol_lines)
    else:
        symbol_block = "- No direct symbol interpretation found in dataset."

    prompt = f"""
Write a concise and supportive dream interpretation.

Requirements:
- Base the interpretation on the user's dream narrative.
- Integrate the emotional/risk indicators provided.
- Use the dataset-based symbol interpretations as grounding.
- Keep it to 2 short sentences maximum.
- Sound natural, reflective, and supportive.
- Do not give direct advice.
- Do not mention AI, model, prompt, dataset, classifier, or pipeline.
- Do not copy the dataset text verbatim for the full answer.
- Summarize and synthesize the meaning clearly.

Dream narrative:
{dream_text}

Risk level:
{stress}

Emotional indicators:
{emotions_text}

Matched symbol interpretations:
{symbol_block}

Final dream interpretation:
""".strip()

    return prompt


def postprocess_interpretation(text):
    text = text.strip()
    prefixes = [
        "final dream interpretation:",
        "dream interpretation:",
        "interpretation:",
        "response:",
        "answer:",
    ]

    lower_text = text.lower()
    for prefix in prefixes:
        if lower_text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    text = re.sub(r"\s+", " ", text).strip()

    if text:
        text = text[0].upper() + text[1:]

    return text


def generate_text(prompt, tokenizer, model, max_new_tokens=100):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
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
    df_interp,
):
    stress = predict_stress(dream_text, stress_tokenizer, stress_model)
    emotions = infer_emotions(dream_text, df_model1, df_symbol_kb)
    matched_interpretations = find_interpretation_matches(dream_text, df_interp, max_matches=5)

    interpretation_prompt = build_dataset_grounded_prompt(
        dream_text=dream_text,
        stress=stress,
        emotions=emotions,
        matched_interpretations=matched_interpretations,
    )

    dream_interpretation = generate_text(
        interpretation_prompt,
        gen_tokenizer,
        gen_model,
        max_new_tokens=100,
    )

    dream_interpretation = postprocess_interpretation(dream_interpretation)

    matched_symbols = [symbol for symbol, _ in matched_interpretations]

    return {
        "dream_text": dream_text,
        "stress_level": stress,
        "emotions": emotions,
        "dream_interpretation": dream_interpretation,
        "matched_symbols": matched_symbols,
    }


try:
    df_model1, df_symbol_kb, df_interp = load_data()
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
                    possible stress signals, emotional patterns, and dataset-grounded dream interpretation.
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
                possible stress signals, emotional patterns, and dataset-grounded dream interpretation.
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
            4. Read the generated dream interpretation grounded in the interpretation dataset.<br>
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
                df_interp=df_interp,
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

        if result["matched_symbols"]:
            st.caption("Matched symbols from interpretation dataset: " + ", ".join(result["matched_symbols"]))
