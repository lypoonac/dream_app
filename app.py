import os
import re
import random
import numpy as np
import pandas as pd
import torch
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Dream Analyzer", page_icon="🌙", layout="wide")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Replace this with your actual Hugging Face model repo
MODEL_REPO = "your-username/dream-stress-classifier"

MODEL1_PATH = os.path.join(DATA_DIR, "model1_train.csv")
MODEL2_PATH = os.path.join(DATA_DIR, "model2_train.csv")
SYMBOL_KB_PATH = os.path.join(DATA_DIR, "symbol_kb.csv")

ID2LABEL = {0: "low", 1: "medium", 2: "high"}


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
    return re.findall(r"[a-zA-Z_]+", text.lower())


@st.cache_data
def load_data():
    df_model1 = pd.read_csv(MODEL1_PATH)
    df_model2 = pd.read_csv(MODEL2_PATH)
    df_symbol_kb = pd.read_csv(SYMBOL_KB_PATH)

    for col in ["dream_text", "stress_label", "emotion_labels", "theme_labels", "symbol_labels"]:
        if col in df_model1.columns:
            df_model1[col] = df_model1[col].apply(clean_text)

    if "stress_label" in df_model1.columns:
        df_model1["stress_label"] = df_model1["stress_label"].str.lower().replace({"moderate": "medium"})
        df_model1 = df_model1[
            df_model1["stress_label"].isin({"low", "medium", "high"})
        ].drop_duplicates().reset_index(drop=True)

    df_model1["emotion_list"] = df_model1["emotion_labels"].apply(split_tags) if "emotion_labels" in df_model1.columns else [[] for _ in range(len(df_model1))]
    df_model1["theme_list"] = df_model1["theme_labels"].apply(split_tags) if "theme_labels" in df_model1.columns else [[] for _ in range(len(df_model1))]
    df_model1["symbol_list"] = df_model1["symbol_labels"].apply(split_tags) if "symbol_labels" in df_model1.columns else [[] for _ in range(len(df_model1))]

    for col in ["stress_label", "emotion_labels", "dominant_emotion", "recommendation_text"]:
        if col in df_model2.columns:
            df_model2[col] = df_model2[col].apply(clean_text)

    if "stress_label" in df_model2.columns:
        df_model2["stress_label"] = df_model2["stress_label"].str.lower().replace({"moderate": "medium"})
        df_model2 = df_model2[
            df_model2["stress_label"].isin({"low", "medium", "high", "very_high", "severe"})
        ].drop_duplicates().reset_index(drop=True)

    df_model2["emotion_list"] = df_model2["emotion_labels"].apply(split_tags) if "emotion_labels" in df_model2.columns else [[] for _ in range(len(df_model2))]

    for col in [
        "symbol_name",
        "traditional_summary_en",
        "theme_tags",
        "emotion_hints",
        "stress_hint",
        "source_origin",
    ]:
        if col in df_symbol_kb.columns:
            df_symbol_kb[col] = df_symbol_kb[col].apply(clean_text)
        else:
            df_symbol_kb[col] = ""

    df_symbol_kb["symbol_name"] = df_symbol_kb["symbol_name"].str.lower()
    df_symbol_kb["theme_list"] = df_symbol_kb["theme_tags"].apply(
        lambda x: [i.strip().lower() for i in clean_text(x).split(",") if i.strip()]
    ) if "theme_tags" in df_symbol_kb.columns else [[] for _ in range(len(df_symbol_kb))]
    df_symbol_kb["emotion_list"] = df_symbol_kb["emotion_hints"].apply(
        lambda x: [i.strip().lower() for i in clean_text(x).split(",") if i.strip()]
    ) if "emotion_hints" in df_symbol_kb.columns else [[] for _ in range(len(df_symbol_kb))]

    symbol_kb_dict = {row["symbol_name"]: row for _, row in df_symbol_kb.iterrows()}

    return df_model1, df_model2, df_symbol_kb, symbol_kb_dict


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).to(DEVICE)
    model.eval()
    return tokenizer, model


df_model1, df_model2, df_symbol_kb, symbol_kb_dict = load_data()
stress_tokenizer, stress_model = load_model()


def predict_stress(text):
    inputs = stress_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
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


def natural_stress_phrase(stress):
    mapping = {
        "low": "fairly calm or reflective",
        "medium": "emotionally unsettled",
        "high": "highly stressed or overwhelmed"
    }
    return mapping.get(stress, "emotionally activated")


def summarize_symbols_naturally(symbols):
    symbol_lines = []

    for sym in symbols[:3]:
        if sym in symbol_kb_dict:
            summary = symbol_kb_dict[sym].get("traditional_summary_en", "")
            summary = str(summary).replace("May suggest ", "").strip()
            summary = summary.rstrip(".")
            symbol_lines.append((sym.replace("_", " "), summary))

    return symbol_lines


def build_interpretation_paragraph(stress, emotions, themes, symbols):
    stress_phrase = natural_stress_phrase(stress)
    intro = f"This dream may reflect a mind that feels {stress_phrase} right now."

    emo_part = (
        f"Feelings like {', '.join(emotions[:3])} seem to be close to the surface."
        if emotions else ""
    )

    symbol_info = summarize_symbols_naturally(symbols)
    if symbol_info:
        symbol_texts = []
        for name, meaning in symbol_info[:2]:
            if meaning:
                symbol_texts.append(f"{name} often point to {meaning}")
            else:
                symbol_texts.append(f"{name} may carry personal symbolic meaning")
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


def build_recommendation_paragraph(stress, emotions, symbols):
    base_rec = retrieve_recommendation(stress, emotions)

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

    symbol_support = ""
    if symbols:
        top_symbol = symbols[0].replace("_", " ")
        symbol_support = f"You may also want to reflect on what the symbol '{top_symbol}' personally means to you right now."

    final_parts = [base_rec, support_line, symbol_support]
    return " ".join([p for p in final_parts if p]).strip()


def analyze_dream(dream_text):
    stress, probs = predict_stress(dream_text)
    emotions, themes, symbols = infer_emotions_themes_symbols(dream_text)

    interpretation = build_interpretation_paragraph(stress, emotions, themes, symbols)
    recommendation = build_recommendation_paragraph(stress, emotions, symbols)

    return {
        "dream_text": dream_text,
        "stress_level": stress,
        "stress_probs": probs,
        "emotions": emotions,
        "themes": themes,
        "symbols": symbols,
        "interpretation": interpretation,
        "recommendation": recommendation
    }


st.title("🌙 Dream Analyzer")
st.write("Pipeline 1 uses a fine-tuned model for stress classification. Pipeline 2 uses retrieval and symbolic interpretation.")

with st.expander("About the pipelines"):
    st.markdown("""
**Pipeline 1 (Fine-tuned):**
- Hugging Face `distilroberta-base`
- fine-tuned for stress classification

**Pipeline 2 (Non-fine-tuned):**
- retrieval from dream examples
- symbol knowledge base lookup
- recommendation retrieval
- natural language response construction
""")

sample_text = "I was late for an exam and everyone stared at me while I forgot everything."
dream_text = st.text_area("Enter your dream", value=sample_text, height=180)

if st.button("Analyze Dream"):
    if dream_text.strip():
        with st.spinner("Analyzing your dream..."):
            result = analyze_dream(dream_text.strip())

        st.success("Analysis completed.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Stress Level")
            st.write(result["stress_level"].upper())

            probs = result["stress_probs"]
            st.write("Confidence scores")
            st.caption(f"Low: {probs[0]:.4f}")
            st.caption(f"Medium: {probs[1]:.4f}")
            st.caption(f"High: {probs[2]:.4f}")

        with col2:
            st.subheader("Detected Features")
            st.write("**Emotions:**", ", ".join(result["emotions"]) if result["emotions"] else "None")
            st.write("**Themes:**", ", ".join(result["themes"]) if result["themes"] else "None")
            st.write("**Symbols:**", ", ".join(result["symbols"]) if result["symbols"] else "None")

        st.subheader("Interpretation")
        st.write(result["interpretation"])

        st.subheader("Recommendation")
        st.write(result["recommendation"])
    else:
        st.warning("Please enter a dream before analysis.")
