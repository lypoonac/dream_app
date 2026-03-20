import os
import re
import random
import numpy as np
import pandas as pd
import torch
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="AI Dream Analyzer for AXA: Early Stress Detection",
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
LOGO_PATH = os.path.join(BASE_DIR, "axa_logo.png")

MODEL1_HF_NAME = "peterjerry111/dream-stress-classifier"
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
        .badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 0.4rem;
            margin-top: 0.3rem;
        }
        .badge-blue {
            background: #e9efff;
            color: #00008F;
        }
        .badge-red {
            background: #ffe9ee;
            color: #d81e3a;
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
    missing_files = []

    if not os.path.exists(MODEL1_PATH):
        missing_files.append(MODEL1_PATH)
    if not os.path.exists(MODEL2_PATH):
        missing_files.append(MODEL2_PATH)
    if not os.path.exists(SYMBOL_KB_PATH):
        missing_files.append(SYMBOL_KB_PATH)

    if missing_files:
        raise FileNotFoundError(
            "Missing required data files:\n" + "\n".join(missing_files)
        )

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
    df_model1["theme_list"] = df_model1["theme_labels"].apply(split_tags)
    df_model1["symbol_list"] = df_model1["symbol_labels"].apply(split_tags)

    for col in ["stress_label", "emotion_labels", "dominant_emotion", "recommendation_text"]:
        if col not in df_model2.columns:
            raise ValueError(f"Column '{col}' missing from model2_train.csv")
        df_model2[col] = df_model2[col].apply(clean_text)

    df_model2["stress_label"] = df_model2["stress_label"].str.lower().replace({"moderate": "medium"})
    df_model2 = df_model2[
        df_model2["stress_label"].isin({"low", "medium", "high", "very_high", "severe"})
    ].drop_duplicates().reset_index(drop=True)
    df_model2["emotion_list"] = df_model2["emotion_labels"].apply(split_tags)

    for col in ["symbol_name", "traditional_summary_en", "theme_tags", "emotion_hints", "stress_hint", "source_origin"]:
        if col not in df_symbol_kb.columns:
            raise ValueError(f"Column '{col}' missing from symbol_kb.csv")
        df_symbol_kb[col] = df_symbol_kb[col].apply(clean_text)

    df_symbol_kb["symbol_name"] = df_symbol_kb["symbol_name"].str.lower()
    df_symbol_kb["theme_list"] = df_symbol_kb["theme_tags"].apply(
        lambda x: [i.strip().lower() for i in clean_text(x).split(",") if i.strip()]
    )
    df_symbol_kb["emotion_list"] = df_symbol_kb["emotion_hints"].apply(
        lambda x: [i.strip().lower() for i in clean_text(x).split(",") if i.strip()]
    )

    symbol_kb_dict = {row["symbol_name"]: row for _, row in df_symbol_kb.iterrows()}

    return df_model1, df_model2, df_symbol_kb, symbol_kb_dict


@st.cache_resource
def load_model1():
    tokenizer = AutoTokenizer.from_pretrained(MODEL1_HF_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL1_HF_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model


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
        probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
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


def natural_stress_phrase(stress):
    mapping = {
        "low": "fairly calm or reflective",
        "medium": "emotionally unsettled",
        "high": "highly stressed or overwhelmed",
    }
    return mapping.get(stress, "emotionally activated")


def summarize_symbols_naturally(symbols, symbol_kb_dict):
    symbol_lines = []

    for sym in symbols[:3]:
        if sym in symbol_kb_dict:
            summary = symbol_kb_dict[sym]["traditional_summary_en"]
            summary = summary.replace("May suggest ", "").strip()
            summary = summary.rstrip(".")
            symbol_lines.append((sym.replace("_", " "), summary))

    return symbol_lines


def build_interpretation_paragraph(stress, emotions, themes, symbols, symbol_kb_dict):
    stress_phrase = natural_stress_phrase(stress)
    intro = f"This dream may reflect a mind that feels {stress_phrase} right now."

    emo_part = (
        f"Feelings like {', '.join(emotions[:3])} seem to be close to the surface."
        if emotions else ""
    )

    symbol_info = summarize_symbols_naturally(symbols, symbol_kb_dict)
    if symbol_info:
        symbol_texts = [f"{name} often point to {meaning}" for name, meaning in symbol_info[:2]]
        symbol_part = "In this kind of dream, " + " and ".join(symbol_texts) + "."
    else:
        symbol_part = ""

    theme_part = (
        f"The overall pattern suggests themes around {', '.join(themes[:3]).replace('_', ' ')}."
        if themes else ""
    )

    closing = (
        "Rather than predicting something literal, the dream is more likely showing how your mind "
        "is processing pressure, change, or unresolved feelings."
    )

    parts = [intro, emo_part, symbol_part, theme_part, closing]
    return " ".join([p for p in parts if p]).strip()


def build_recommendation_paragraph(stress, emotions, df_model2):
    base_rec = retrieve_recommendation(stress, emotions, df_model2)

    if stress == "high":
        support_line = (
            "If this dream matches how you have been feeling lately, it may help to slow down, "
            "reduce stimulation, and focus only on the next manageable step."
        )
    elif stress == "medium":
        support_line = (
            "It may help to simplify your day, check in with your emotions, and avoid forcing clarity too quickly."
        )
    else:
        support_line = (
            "This can be a good moment to keep things steady, listen to yourself, and move gently with what feels meaningful."
        )

    return f"{base_rec} {support_line}".strip()


def analyze_dream(dream_text, tokenizer, model, df_model1, df_model2, df_symbol_kb, symbol_kb_dict):
    stress, probs = predict_stress(dream_text, tokenizer, model)
    emotions, themes, symbols = infer_emotions_themes_symbols(dream_text, df_model1, df_symbol_kb)

    interpretation = build_interpretation_paragraph(
        stress=stress,
        emotions=emotions,
        themes=themes,
        symbols=symbols,
        symbol_kb_dict=symbol_kb_dict,
    )

    recommendation = build_recommendation_paragraph(
        stress=stress,
        emotions=emotions,
        df_model2=df_model2,
    )

    return {
        "dream_text": dream_text,
        "stress_level": stress,
        "stress_probs": probs,
        "emotions": emotions,
        "themes": themes,
        "symbols": symbols,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


try:
    df_model1, df_model2, df_symbol_kb, symbol_kb_dict = load_data()
    stress_tokenizer, stress_model = load_model1()
except Exception as e:
    st.error("Failed to load required files or model.")
    st.exception(e)
    st.stop()


header_col1, header_col2 = st.columns([1, 4])

with header_col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=140)

with header_col2:
    st.markdown(
        """
        <div class="brand-card">
            <div class="main-title">AI Dream Analyzer for AXA: Early Stress Detection</div>
            <div class="sub-title">
                A prototype wellness support tool combining transformer-based stress detection
                with symbolic dream interpretation and recommendation retrieval.
            </div>
            <span class="badge badge-blue">Model 1: Hugging Face Stress Classifier</span>
            <span class="badge badge-red">Model 2: Symbolic Interpretation Pipeline</span>
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
            3. Review the predicted stress level, detected emotions, themes, and symbols.<br>
            4. Read the generated interpretation and recommendation for early reflection.<br>
            5. This tool is for wellness support and early awareness, not medical diagnosis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("About the two models"):
    st.markdown(
        """
**Model 1 — Stress Detection**
- Hugging Face model: `peterjerry111/dream-stress-classifier`
- Predicts stress level: `low`, `medium`, `high`

**Model 2 — Interpretation Pipeline**
- Retrieval from similar dream examples
- Symbol knowledge base lookup
- Emotion/theme/symbol inference
- Recommendation retrieval
- Natural language paragraph construction

**Note**
- This app is designed for reflective and supportive use.
- It is not a substitute for professional mental health care.
        """
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
                tokenizer=stress_tokenizer,
                model=stress_model,
                df_model1=df_model1,
                df_model2=df_model2,
                df_symbol_kb=df_symbol_kb,
                symbol_kb_dict=symbol_kb_dict,
            )

        st.success("Analysis completed.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Stress Level")
            st.write(result["stress_level"].upper())

            probs = result["stress_probs"]
            st.write("Confidence scores")
            st.caption(f"Low: {probs[0]:.4f}")
            st.caption(f"Medium: {probs[1]:.4f}")
            st.caption(f"High: {probs[2]:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Detected Features")
            st.write("**Emotions:**", ", ".join(result["emotions"]) if result["emotions"] else "None")
            st.write("**Themes:**", ", ".join(result["themes"]) if result["themes"] else "None")
            st.write("**Symbols:**", ", ".join(result["symbols"]) if result["symbols"] else "None")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Interpretation")
        st.write(result["interpretation"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Recommendation")
        st.write(result["recommendation"])
        st.markdown("</div>", unsafe_allow_html=True)
