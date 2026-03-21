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

DREAM_INTERPRETATION_CANDIDATES = [
    os.path.join(DATA_DIR, "dreams_interpretations_dataset.csv"),
    os.path.join(DATA_DIR, "dreams_interpretations.csv"),
]

MODEL1_HF_NAME = "peterjerry111/dream-stress-classifier"

# Optional enrichment model
# If this repo is broken/unavailable, app will safely fall back to retrieval-only
MODEL2_HF_NAME = "peterjerry111/dream_interpretation_model"

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
            padding: 1.2rem;
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
    return re.findall(r"[a-zA-Z]+", str(text).lower())


def normalize_symbol(text):
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s/&'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text):
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_first_sentence(text):
    sentences = split_into_sentences(text)
    return sentences[0] if sentences else clean_text(text)


def find_existing_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


@st.cache_data
def load_data():
    interp_path = find_existing_file(DREAM_INTERPRETATION_CANDIDATES)

    missing = []
    if not os.path.exists(MODEL1_PATH):
        missing.append(MODEL1_PATH)
    if not os.path.exists(SYMBOL_KB_PATH):
        missing.append(SYMBOL_KB_PATH)
    if interp_path is None:
        missing.extend(DREAM_INTERPRETATION_CANDIDATES)

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    df_model1 = pd.read_csv(MODEL1_PATH)
    df_symbol_kb = pd.read_csv(SYMBOL_KB_PATH)
    df_interp = pd.read_csv(interp_path)

    for col in ["dream_text", "stress_label", "emotion_labels", "theme_labels", "symbol_labels"]:
        if col not in df_model1.columns:
            raise ValueError(f"Column '{col}' missing from model1_train.csv")
        df_model1[col] = df_model1[col].apply(clean_text)

    df_model1["stress_label"] = df_model1["stress_label"].str.lower().replace({"moderate": "medium"})
    df_model1 = df_model1[df_model1["stress_label"].isin({"low", "medium", "high"})].drop_duplicates().reset_index(drop=True)
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
        raise ValueError("Interpretation CSV must contain 'Dream Symbol' and 'Interpretation' columns")

    df_interp = df_interp[[interp_symbol_col, interp_text_col]].copy()
    df_interp.columns = ["dream_symbol", "interpretation"]
    df_interp["dream_symbol"] = df_interp["dream_symbol"].apply(clean_text)
    df_interp["interpretation"] = df_interp["interpretation"].apply(clean_text)
    df_interp = df_interp[(df_interp["dream_symbol"] != "") & (df_interp["interpretation"] != "")]
    df_interp["symbol_norm"] = df_interp["dream_symbol"].apply(normalize_symbol)
    df_interp = df_interp.drop_duplicates(subset=["symbol_norm", "interpretation"]).reset_index(drop=True)

    return df_model1, df_symbol_kb, df_interp, interp_path


@st.cache_resource
def load_stress_model():
    stress_tokenizer = AutoTokenizer.from_pretrained(MODEL1_HF_NAME)
    stress_model = AutoModelForSequenceClassification.from_pretrained(MODEL1_HF_NAME).to(DEVICE)
    stress_model.eval()
    return stress_tokenizer, stress_model


@st.cache_resource
def load_enrichment_model():
    try:
        gen_tokenizer = AutoTokenizer.from_pretrained(MODEL2_HF_NAME)
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL2_HF_NAME).to(DEVICE)
        gen_model.eval()
        return gen_tokenizer, gen_model, True, ""
    except Exception as e:
        return None, None, False, str(e)


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

    dream_tokens = set(tokenize_simple(text))
    direct_symbol_rows = []
    for _, row in df_symbol_kb.iterrows():
        symbol = clean_text(row["symbol_name"]).lower()
        symbol_parts = symbol.split("_")
        if all(part in dream_tokens for part in symbol_parts):
            direct_symbol_rows.append(row)

    if direct_symbol_rows:
        symbol_emotions = []
        for row in direct_symbol_rows:
            symbol_emotions.extend(row["emotion_list"])

        if symbol_emotions:
            combined = emotions + symbol_emotions
            emotions = pd.Series(combined).value_counts().head(5).index.tolist()

    return emotions


def find_interpretation_matches(dream_text, df_interp, max_matches=10):
    dream_text_norm = normalize_symbol(dream_text)
    dream_tokens = set(tokenize_simple(dream_text_norm))
    matches = []

    for _, row in df_interp.iterrows():
        symbol = clean_text(row["dream_symbol"])
        interpretation = clean_text(row["interpretation"])
        symbol_norm = normalize_symbol(symbol)

        if not symbol_norm:
            continue

        symbol_tokens = set(tokenize_simple(symbol_norm))
        score = 0

        if f" {symbol_norm} " in f" {dream_text_norm} ":
            score += 10

        overlap = len(symbol_tokens & dream_tokens)
        score += overlap * 3

        if len(symbol_tokens) == 1 and list(symbol_tokens)[0] in dream_tokens:
            score += 5

        if score > 0:
            matches.append((score, symbol, interpretation))

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


def get_random_interpretation(df_interp):
    row = df_interp.sample(1).iloc[0]
    return {
        "matched_symbols": [clean_text(row["dream_symbol"])],
        "retrieved_interpretation": clean_text(row["interpretation"]),
        "random_used": True,
    }


def get_best_dataset_interpretation(dream_text, df_interp):
    matches = find_interpretation_matches(dream_text, df_interp, max_matches=10)

    if not matches:
        return get_random_interpretation(df_interp)

    best_symbol, best_interpretation = matches[0]
    return {
        "matched_symbols": [s for s, _ in matches[:5]],
        "retrieved_interpretation": best_interpretation,
        "random_used": False,
    }


def build_enrichment_prompt(dream_text, retrieved_interpretation):
    return (
        "Rewrite the following dream interpretation into a short, clear, natural explanation. "
        "Keep it grounded in the reference interpretation and make it relevant to the dream. "
        "Do not repeat the prompt. Do not mention 'reference interpretation'.\n\n"
        f"Dream: {dream_text}\n\n"
        f"Reference interpretation: {retrieved_interpretation}\n\n"
        "Improved interpretation:"
    )


def generate_enriched_interpretation(dream_text, retrieved_interpretation, tokenizer, model):
    prompt = build_enrichment_prompt(dream_text, retrieved_interpretation)

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
            max_new_tokens=120,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def is_weird_enrichment(output_text, dream_text, retrieved_interpretation):
    out = clean_text(output_text)
    if not out:
        return True

    out_low = out.lower()
    dream_low = clean_text(dream_text).lower()
    ref_low = clean_text(retrieved_interpretation).lower()

    if len(out) < 30:
        return True

    bad_starts = [
        "interpret dream symbol",
        "dream symbol",
        "improved interpretation:",
        "reference interpretation:",
        "dream:",
    ]
    if any(out_low.startswith(x) for x in bad_starts):
        return True

    if out_low == dream_low:
        return True

    if out_low == ref_low:
        return False

    if "interpret dream" in out_low:
        return True

    unique_words = set(re.findall(r"[a-zA-Z]+", out_low))
    if len(unique_words) < 6:
        return True

    return False


def hybrid_interpretation(dream_text, retrieved_interpretation, enrich_tokenizer, enrich_model, enrich_available):
    if not enrich_available or enrich_tokenizer is None or enrich_model is None:
        return retrieved_interpretation, False, "Enrichment model unavailable"

    try:
        enriched = generate_enriched_interpretation(
            dream_text,
            retrieved_interpretation,
            enrich_tokenizer,
            enrich_model
        )

        if is_weird_enrichment(enriched, dream_text, retrieved_interpretation):
            return retrieved_interpretation, False, "Enrichment output looked invalid"

        return enriched, True, ""
    except Exception as e:
        return retrieved_interpretation, False, str(e)


def analyze_dream(
    dream_text,
    stress_tokenizer,
    stress_model,
    enrich_tokenizer,
    enrich_model,
    enrich_available,
    df_model1,
    df_symbol_kb,
    df_interp,
):
    stress = predict_stress(dream_text, stress_tokenizer, stress_model)
    emotions = infer_emotions(dream_text, df_model1, df_symbol_kb)

    retrieval = get_best_dataset_interpretation(dream_text, df_interp)
    retrieved_interpretation = retrieval["retrieved_interpretation"]

    final_interpretation, enriched_used, enrich_note = hybrid_interpretation(
        dream_text,
        retrieved_interpretation,
        enrich_tokenizer,
        enrich_model,
        enrich_available
    )

    return {
        "stress_level": stress,
        "emotions": emotions,
        "dream_interpretation": final_interpretation,
        "retrieved_interpretation": retrieved_interpretation,
        "matched_symbols": retrieval["matched_symbols"],
        "random_used": retrieval["random_used"],
        "enriched_used": enriched_used,
        "enrich_note": enrich_note,
    }


try:
    df_model1, df_symbol_kb, df_interp, interp_path_used = load_data()
    stress_tokenizer, stress_model = load_stress_model()
    enrich_tokenizer, enrich_model, enrich_available, enrich_error = load_enrichment_model()
except Exception as e:
    st.error("Failed to load required files or models.")
    st.exception(e)
    st.stop()


if os.path.exists(LOGO_PATH):
    c1, c2 = st.columns([1, 4])
    with c1:
        st.image(LOGO_PATH, width=140)
    with c2:
        st.markdown(
            """
            <div class="brand-card">
                <div class="main-title">AXA AI Dream Analyzer: Early Stress Detection</div>
                <div class="sub-title">
                    A hybrid prototype that combines retrieval-based interpretation with optional model enrichment.
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
                A hybrid prototype that combines retrieval-based interpretation with optional model enrichment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="guide-card">
        <div class="section-title">How this version works</div>
        <div class="small-muted">
            1. Your dream is analyzed for stress level and emotions.<br>
            2. The app retrieves the closest dream interpretation from the dataset.<br>
            3. If the enrichment model behaves well, it rewrites the interpretation more naturally.<br>
            4. If the model output looks strange, the app safely falls back to the retrieved dataset interpretation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not enrich_available:
    st.warning("Enrichment model could not be loaded. The app will use retrieval-only interpretation.")

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
                dream_text.strip(),
                stress_tokenizer,
                stress_model,
                enrich_tokenizer,
                enrich_model,
                enrich_available,
                df_model1,
                df_symbol_kb,
                df_interp,
            )

        st.success("Analysis completed.")

        stress_text = html.escape(result["stress_level"].upper())
        emotions_text = html.escape(", ".join(result["emotions"]) if result["emotions"] else "None")
        interpretation_text = html.escape(result["dream_interpretation"])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Stress Level")
            st.markdown(f"<div class='highlight-text'>{stress_text}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Emotions")
            st.markdown(f"<div class='highlight-text'>{emotions_text}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Dream Interpretation")
        st.markdown(f"<div class='highlight-block'>{interpretation_text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if result["random_used"]:
            st.caption("No direct interpretation match found. A random dataset interpretation was used as the base.")
        else:
            st.caption("Matched symbols: " + ", ".join(result["matched_symbols"]))

        if result["enriched_used"]:
            st.caption("Interpretation was enriched by the model.")
        else:
            st.caption("Interpretation shown is the retrieved dataset interpretation.")

        with st.expander("Show retrieval details"):
            st.write("**Retrieved dataset interpretation:**")
            st.write(result["retrieved_interpretation"])
            if result["enrich_note"]:
                st.write("**Enrichment note:**")
                st.write(result["enrich_note"])
