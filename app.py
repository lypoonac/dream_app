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


def build_interpretation_prompt(dream_text, stress, emotions, themes, symbols, symbol_kb_dict):
    emotions_text = ", ".join(emotions[:5]) if emotions else "none clearly inferred"
    themes_text = ", ".join([t.replace("_", " ") for t in themes[:5]]) if themes else "none clearly inferred"
    symbols_text = ", ".join([s.replace("_", " ") for s in symbols[:5]]) if symbols else "none clearly inferred"

    symbol_meanings = []
    for sym in symbols[:3]:
        if sym in symbol_kb_dict:
            meaning = clean_text(symbol_kb_dict[sym]["traditional_summary_en"])
            if meaning:
                symbol_meanings.append(f"{sym.replace('_', ' ')}: {meaning}")

    symbol_meanings_text = " | ".join(symbol_meanings) if symbol_meanings else "no extra symbol meanings"

    prompt = f"""
Answer in 3 to 5 sentences only.

Task: Write a short dream interpretation in clear, natural English.

Rules:
- Be calm, supportive, and human-friendly.
- Do not repeat the instructions.
- Do not copy the prompt.
- Do not say "Task", "Rules", or "Dream text".
- Do not predict the future.
- Do not claim supernatural certainty.
- Focus on psychological and symbolic meaning.

Dream text: {dream_text}
Predicted stress level: {stress}
Inferred emotions: {emotions_text}
Inferred themes: {themes_text}
Inferred symbols: {symbols_text}
Symbol meanings: {symbol_meanings_text}

Interpretation:
""".strip()
    return prompt


def build_recommendation_prompt(dream_text, stress, emotions, themes, symbols, retrieved_recommendation):
    emotions_text = ", ".join(emotions[:5]) if emotions else "none clearly inferred"
    themes_text = ", ".join([t.replace("_", " ") for t in themes[:5]]) if themes else "none clearly inferred"
    symbols_text = ", ".join([s.replace("_", " ") for s in symbols[:5]]) if symbols else "none clearly inferred"

    prompt = f"""
Answer in 2 to 4 sentences only.

Task: Write a short supportive recommendation based on the dream analysis.

Rules:
- Be gentle, practical, and emotionally supportive.
- Do not repeat the instructions.
- Do not copy the prompt.
- Do not give medical, legal, or dangerous advice.
- Do not mention AI or model names.
- Use the suggested recommendation naturally if it fits.

Dream text: {dream_text}
Predicted stress level: {stress}
Inferred emotions: {emotions_text}
Inferred themes: {themes_text}
Inferred symbols: {symbols_text}
Suggested recommendation: {retrieved_recommendation}

Recommendation:
""".strip()
    return prompt


def generate_text(prompt, tokenizer, model, max_new_tokens=120):
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
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    for prefix in ["Interpretation:", "Recommendation:", "Answer:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text


def analyze_dream(
    dream_text,
    stress_tokenizer,
    stress_model,
    gen_tokenizer,
    gen_model,
    df_model1,
    df_model2,
    df_symbol_kb,
    symbol_kb_dict,
):
    stress, probs = predict_stress(dream_text, stress_tokenizer, stress_model)
    emotions, themes, symbols = infer_emotions_themes_symbols(
        dream_text, df_model1, df_symbol_kb
    )

    retrieved_recommendation = retrieve_recommendation(stress, emotions, df_model2)

    interpretation_prompt = build_interpretation_prompt(
        dream_text=dream_text,
        stress=stress,
        emotions=emotions,
        themes=themes,
        symbols=symbols,
        symbol_kb_dict=symbol_kb_dict,
    )

    recommendation_prompt = build_recommendation_prompt(
        dream_text=dream_text,
        stress=stress,
        emotions=emotions,
        themes=themes,
        symbols=symbols,
        retrieved_recommendation=retrieved_recommendation,
    )

    interpretation = generate_text(
        interpretation_prompt,
        gen_tokenizer,
        gen_model,
        max_new_tokens=120,
    )

    recommendation = generate_text(
        recommendation_prompt,
        gen_tokenizer,
        gen_model,
        max_new_tokens=100,
    )

    return {
        "dream_text": dream_text,
        "stress_level": stress,
        "stress_probs": probs,
        "emotions": emotions,
        "themes": themes,
        "symbols": symbols,
        "retrieved_recommendation": retrieved_recommendation,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


try:
    df_model1, df_model2, df_symbol_kb, symbol_kb_dict = load_data()
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
                <div class="main-title">AI Dream Analyzer for AXA: Early Stress Detection</div>
                <div class="sub-title">
                    A prototype wellness support tool combining transformer-based stress detection
                    with Hugging Face interpretation generation.
                </div>
                <span class="badge badge-blue">Model 1: Stress Classifier</span>
                <span class="badge badge-red">Model 2: FLAN-T5 Generator</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div class="brand-card">
            <div class="main-title">AI Dream Analyzer for AXA: Early Stress Detection</div>
            <div class="sub-title">
                A prototype wellness support tool combining transformer-based stress detection
                with Hugging Face interpretation generation.
            </div>
            <span class="badge badge-blue">Model 1: Stress Classifier</span>
            <span class="badge badge-red">Model 2: FLAN-T5 Generator</span>
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
            3. Review the predicted stress level, inferred emotions, themes, and symbols.<br>
            4. Read the generated interpretation and recommendation.<br>
            5. This tool supports reflection and early stress awareness, not diagnosis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("About the two models"):
    st.markdown(
        f"""
**Model 1 — Stress Detection**
- Hugging Face model: `{MODEL1_HF_NAME}`
- Predicts stress level: `low`, `medium`, `high`

**Model 2 — Interpretation and Recommendation Generation**
- Hugging Face model: `{MODEL2_HF_NAME}`
- Generates interpretation and recommendation text from structured prompts

**Support Data**
- `model1_train.csv`
- `model2_train.csv`
- `symbol_kb.csv`

**Note**
- This app is for supportive reflection only.
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
                stress_tokenizer=stress_tokenizer,
                stress_model=stress_model,
                gen_tokenizer=gen_tokenizer,
                gen_model=gen_model,
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
            st.subheader("Inferred Features")
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

        with st.expander("Model 2 supporting context"):
            st.write("**Retrieved recommendation seed:**", result["retrieved_recommendation"])
            st.write("**Model 2:**", MODEL2_HF_NAME)
