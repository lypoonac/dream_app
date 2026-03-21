import os
import re
import random
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
    page_title="AXA AI Dream Analyzer - Early Stress Detection",
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
            background: #fff7cc;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            border: 1px solid #f2d66b;
            box-shadow: 0 4px 16px rgba(160, 120, 0, 0.08);
            margin-bottom: 1rem;
        }

        .normal-card {
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
    df_model1 = df_model1[
        df_model1["stress_label"].isin({"low", "medium", "high"})
    ].drop_duplicates().reset_index(drop=True)
    df_model1["emotion_list"] = df_model1["emotion_labels"].apply(split_tags)
    df_model1["theme_list"] = df_model1["theme_labels"].apply(split_tags)
    df_model1["symbol_list"] = df_model1["symbol_labels"].apply(split_tags)

    for col in ["stress_label", "emotion_labels", "dominant_emotion", "recommendation_text"]:
        df_model2[col] = df_model2[col].apply(clean_text)

    df_model2["stress_label"] = df_model2["stress_label"].str.lower().replace({"moderate": "medium"})
    df_model2 = df_model2[
        df_model2["stress_label"].isin({"low", "medium", "high", "very_high", "severe"})
    ].drop_duplicates().reset_index(drop=True)
    df_model2["emotion_list"] = df_model2["emotion_labels"].apply(split_tags)

    for col in ["symbol_name", "traditional_summary_en", "theme_tags", "emotion_hints", "stress_hint", "source_origin"]:
        df_symbol_kb[col] = df_symbol_kb[col].apply(clean_text)

    df_symbol_kb["symbol_name"] = df_symbol_kb["symbol_name"].str.lower()
    df_symbol_kb["theme_list"] = df_symbol_kb["theme_tags"].apply(split_tags)
    df_symbol_kb["emotion_list"] = df_symbol_kb["emotion_hints"].apply(split_tags)

    symbol_kb_dict = {row["symbol_name"]: row for _, row in df_symbol_kb.iterrows()}

    return df_model1, df_model2, df_symbol_kb, symbol_kb_dict


@st.cache_resource
def load_models():
    stress_tokenizer = AutoTokenizer.from_pretrained(MODEL1_HF_NAME)
    stress_model = AutoModelForSequenceClassification.from_pretrained(MODEL1_HF_NAME).to(DEVICE)
    stress_model.eval()

    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE)
    gen_model.eval()

    return stress_tokenizer, stress_model, gen_tokenizer, gen_model


df_model1, df_model2, df_symbol_kb, symbol_kb_dict = load_data()
stress_tokenizer, stress_model, gen_tokenizer, gen_model = load_models()


def predict_stress(text):
    inputs = stress_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = stress_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

    return ID2LABEL[pred], probs


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


def find_similar_examples(text, top_k=5):
    query_tokens = set(tokenize_simple(text))
    scores = []

    for _, row in df_model1.iterrows():
        row_tokens = set(tokenize_simple(row["dream_text"]))
        overlap = len(query_tokens & row_tokens)
        if overlap > 0:
            scores.append((overlap, row))

    scores = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return [row for _, row in scores]


def infer_emotions_themes_symbols(text):
    matches = find_similar_examples(text, top_k=5)

    emotions = []
    themes = []
    symbols = []

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


def retrieve_recommendation(pred_stress, inferred_emotions):
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


def summarize_symbols_naturally(symbols):
    symbol_lines = []

    for sym in symbols[:3]:
        if sym in symbol_kb_dict:
            summary = clean_text(symbol_kb_dict[sym]["traditional_summary_en"])
            summary = summary.replace("May suggest ", "").strip()
            summary = summary.rstrip(".")
            symbol_lines.append((sym.replace("_", " "), summary))

    return symbol_lines


def build_interpretation_prompt(dream_text, stress, emotions, themes, symbols):
    emotion_text = ", ".join(emotions[:5]) if emotions else "none clearly detected"
    theme_text = ", ".join([t.replace("_", " ") for t in themes[:5]]) if themes else "none clearly detected"
    symbol_text = ", ".join([s.replace("_", " ") for s in symbols[:5]]) if symbols else "none clearly detected"

    symbol_summaries = summarize_symbols_naturally(symbols)
    if symbol_summaries:
        kb_text = "; ".join([f"{name}: {meaning}" for name, meaning in symbol_summaries[:3]])
    else:
        kb_text = "none"

    return f"""
You are a supportive dream reflection assistant.

Dream text:
{dream_text}

Predicted stress level:
{stress}

Detected emotions:
{emotion_text}

Detected themes:
{theme_text}

Detected symbols:
{symbol_text}

Symbol meanings from knowledge base:
{kb_text}

Write one short interpretation paragraph in 4 to 5 sentences.
Keep it calm, reflective, and non-clinical.
Do not claim certainty.
Do not say the dream predicts the future.
Explain that the dream may reflect the person's emotional processing.
""".strip()


def build_wellbeing_tips_prompt(dream_text, stress, emotions, themes, symbols, fallback_tip):
    emotion_text = ", ".join(emotions[:5]) if emotions else "none clearly detected"
    theme_text = ", ".join([t.replace("_", " ") for t in themes[:5]]) if themes else "none clearly detected"
    symbol_text = ", ".join([s.replace("_", " ") for s in symbols[:5]]) if symbols else "none clearly detected"

    return f"""
You are a supportive well-being assistant.

Dream text:
{dream_text}

Predicted stress level:
{stress}

Detected emotions:
{emotion_text}

Detected themes:
{theme_text}

Detected symbols:
{symbol_text}

Fallback support tip:
{fallback_tip}

Write well-being tips in 3 to 4 sentences.
Keep the tone gentle, practical, and supportive.
Focus on rest, reflection, emotional balance, and manageable next steps.
Do not mention diagnosis.
Do not mention mental illness.
""".strip()


def generate_text(prompt, tokenizer, model, max_new_tokens=120):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def is_bad_generated_text(text):
    if not text:
        return True

    lowered = text.lower().strip()

    bad_prefixes = [
        "dream text:",
        "predicted stress level:",
        "detected emotions:",
        "detected themes:",
        "detected symbols:",
        "fallback support tip:",
        "you are a supportive",
        "write one short interpretation",
        "write well-being tips",
    ]

    if any(lowered.startswith(prefix) for prefix in bad_prefixes):
        return True

    if len(text.split()) < 8:
        return True

    return False


def analyze_dream(dream_text):
    stress, probs = predict_stress(dream_text)
    emotions, themes, symbols = infer_emotions_themes_symbols(dream_text)

    interpretation_prompt = build_interpretation_prompt(
        dream_text=dream_text,
        stress=stress,
        emotions=emotions,
        themes=themes,
        symbols=symbols,
    )
    interpretation = generate_text(
        interpretation_prompt,
        gen_tokenizer,
        gen_model,
        max_new_tokens=140,
    )

    fallback_tip = retrieve_recommendation(stress, emotions)

    wellbeing_prompt = build_wellbeing_tips_prompt(
        dream_text=dream_text,
        stress=stress,
        emotions=emotions,
        themes=themes,
        symbols=symbols,
        fallback_tip=fallback_tip,
    )
    wellbeing_tips = generate_text(
        wellbeing_prompt,
        gen_tokenizer,
        gen_model,
        max_new_tokens=120,
    )

    if is_bad_generated_text(interpretation):
        interpretation = (
            "This dream may reflect how your mind is processing recent emotions, pressure, "
            "or unresolved thoughts. The dream content may be a symbolic expression of your "
            "current inner state rather than something literal. It can be helpful to view it "
            "as a reflection of emotional processing and personal stress."
        )

    if is_bad_generated_text(wellbeing_tips):
        wellbeing_tips = fallback_tip

    combined_wellbeing = f"{wellbeing_tips}\n\n{interpretation}"

    return {
        "dream_text": dream_text,
        "stress_level": stress,
        "stress_probs": probs,
        "emotions": emotions,
        "themes": themes,
        "symbols": symbols,
        "interpretation": interpretation,
        "wellbeing_tips": wellbeing_tips,
        "combined_wellbeing": combined_wellbeing,
    }


logo_path = os.path.join(BASE_DIR, "axa_logo.png")

header_col1, header_col2 = st.columns([1, 4])

with header_col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=140)

with header_col2:
    st.markdown(
        """
        <div class="brand-card">
            <div class="main-title">AXA AI Dream Analyzer - Early Stress Detection</div>
            <div class="sub-title">
                A prototype wellness support tool for reflective dream analysis and early stress awareness.
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
            3. Review the predicted stress level and detected emotions.<br>
            4. Read the <b>Well-being tips</b> for supportive reflection.<br>
            5. This tool is for wellness support and early awareness, not medical diagnosis.
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
    if dream_text.strip():
        with st.spinner("Analyzing your dream..."):
            result = analyze_dream(dream_text.strip())

        st.success("Analysis completed.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="normal-card">', unsafe_allow_html=True)
            st.subheader("Stress Level")
            st.write(result["stress_level"].upper())
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="normal-card">', unsafe_allow_html=True)
            st.subheader("Detected Emotions")
            st.write(", ".join(result["emotions"]) if result["emotions"] else "None")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Well-being tips")
        st.write(result["combined_wellbeing"])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a dream before analysis.")
