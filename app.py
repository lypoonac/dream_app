# Standard library imports
import os
import re
import random
import base64

# Third-party data and ML libraries
import numpy as np
import pandas as pd
import torch
import streamlit as st

# Hugging Face Transformers for classification and text generation
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

# Configure the Streamlit page
st.set_page_config(
    page_title="AXA Health Dream Analyzer",
    page_icon="🌙",
    layout="wide",
)

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# If GPU is available, also seed CUDA operations
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Select device: GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define base directory and data folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Define file paths for datasets
MODEL1_PATH = os.path.join(DATA_DIR, "model1_train.csv")
MODEL2_PATH = os.path.join(DATA_DIR, "model2_train.csv")
SYMBOL_KB_PATH = os.path.join(DATA_DIR, "symbol_kb.csv")

# Hugging Face model names
MODEL1_HF_NAME = "peterjerry111/dream-stress-classifier"
GEN_MODEL_NAME = "google/flan-t5-base"

# Mapping model output IDs to readable stress labels
ID2LABEL = {0: "low", 1: "medium", 2: "high"}

# Custom CSS styling for the Streamlit app UI
st.markdown(
    """
    <style>
        :root {
            --axa-blue: #00008f;
            --axa-blue-deep: #05055f;
            --axa-blue-soft: #e9edff;
            --axa-red: #ff3b43;
            --bg-main: #f4f7fc;
            --bg-alt: #eef3fb;
            --card-bg: rgba(255, 255, 255, 0.96);
            --text-main: #14213d;
            --text-soft: #5b6785;
            --border-soft: #d9e2f2;
            --shadow-soft: 0 10px 30px rgba(6, 24, 70, 0.08);
            --shadow-strong: 0 18px 40px rgba(6, 24, 70, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(0, 0, 143, 0.08), transparent 28%),
                linear-gradient(180deg, #f8fbff 0%, #eef3fb 100%);
            color: var(--text-main);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .app-shell {
            padding: 0.25rem 0 1.5rem 0;
        }

        .hero-wrap {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, var(--axa-blue) 0%, #101e9c 60%, #1b2bb3 100%);
            border-radius: 24px;
            padding: 2rem 2rem 1.75rem 2rem;
            box-shadow: var(--shadow-strong);
            margin-bottom: 1.25rem;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .hero-wrap::before {
            content: "";
            position: absolute;
            top: -30px;
            right: -20px;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(255,255,255,0.16), transparent 65%);
            border-radius: 50%;
        }

        .hero-wrap::after {
            content: "";
            position: absolute;
            top: 18px;
            right: 82px;
            width: 10px;
            height: 110px;
            background: var(--axa-red);
            transform: rotate(35deg);
            border-radius: 999px;
            box-shadow: 0 0 16px rgba(255, 59, 67, 0.35);
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 110px 1fr;
            gap: 1.25rem;
            align-items: center;
            position: relative;
            z-index: 2;
        }

        .hero-logo-box {
            width: 96px;
            height: 96px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 20px;
            backdrop-filter: blur(6px);
        }

        .hero-logo-box img {
            max-width: 72px;
        }

        .hero-title {
            color: #ffffff;
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.12;
            margin: 0 0 0.45rem 0;
        }

        .hero-subtitle {
            color: rgba(255,255,255,0.88);
            font-size: 1rem;
            line-height: 1.6;
            max-width: 760px;
            margin: 0;
        }

        .panel-card {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 20px;
            box-shadow: var(--shadow-soft);
            padding: 1.25rem 1.25rem;
            margin-bottom: 1rem;
        }

        .panel-card.soft-blue {
            background: linear-gradient(180deg, #ffffff 0%, #f8faff 100%);
        }

        .section-title {
            color: var(--axa-blue);
            font-size: 1.02rem;
            font-weight: 800;
            margin-bottom: 0.6rem;
            letter-spacing: 0.01em;
        }

        .section-title.with-accent {
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .section-title.with-accent::before {
            content: "";
            width: 6px;
            height: 22px;
            border-radius: 999px;
            background: linear-gradient(180deg, var(--axa-red), #ff7a80);
        }

        .guide-text {
            color: var(--text-soft);
            font-size: 0.97rem;
            line-height: 1.8;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            padding: 1.1rem 1.15rem;
            box-shadow: var(--shadow-soft);
            min-height: 150px;
        }

        .metric-label {
            color: var(--text-soft);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }

        .metric-value {
            color: var(--axa-blue);
            font-size: 1.65rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            line-height: 1.2;
        }

        .metric-helper {
            color: var(--text-soft);
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .tips-card {
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            border: 1px solid var(--border-soft);
            border-radius: 20px;
            box-shadow: var(--shadow-soft);
            padding: 1.2rem 1.25rem;
            margin-top: 0.25rem;
            position: relative;
            overflow: hidden;
        }

        .tips-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--axa-blue), var(--axa-red));
        }

        .tips-text {
            color: var(--axa-blue);
            font-size: 1rem;
            line-height: 1.9;
            margin-top: 0.2rem;
        }

        .highlight-text {
            background: linear-gradient(transparent 58%, rgba(255, 255, 255, 0.95) 58%);
            color: var(--axa-blue);
            font-weight: 700;
            padding: 0 0.08rem;
            display: inline;
        }

        .stTextArea label {
            color: var(--axa-blue) !important;
            font-weight: 700 !important;
        }

        .stTextArea textarea {
            background: #ffffff !important;
            color: var(--text-main) !important;
            border: 1px solid #cfd9ec !important;
            border-radius: 16px !important;
            padding: 1rem !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.03);
        }

        .stTextArea textarea:focus {
            border: 1px solid var(--axa-blue) !important;
            box-shadow: 0 0 0 3px rgba(0, 0, 143, 0.10) !important;
        }

        div.stButton > button {
            width: 100%;
            background: linear-gradient(135deg, var(--axa-blue) 0%, #2337c6 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.82rem 1.2rem;
            font-weight: 800;
            font-size: 0.98rem;
            letter-spacing: 0.01em;
            box-shadow: 0 10px 24px rgba(0, 0, 143, 0.22);
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
            background: linear-gradient(135deg, #09097c 0%, #2b43de 100%);
            box-shadow: 0 14px 26px rgba(0, 0, 143, 0.28);
            color: white;
        }

        div.stButton > button:focus:not(:active) {
            border: none;
            box-shadow: 0 0 0 4px rgba(0, 0, 143, 0.16);
            color: white;
        }

        .footer-note {
            color: var(--text-soft);
            font-size: 0.88rem;
            line-height: 1.6;
            text-align: center;
            margin-top: 1rem;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.4rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            background: var(--axa-blue-soft);
            color: var(--axa-blue);
            font-size: 0.88rem;
            font-weight: 700;
            border: 1px solid #d6def8;
        }

        @media (max-width: 860px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .hero-logo-box {
                width: 84px;
                height: 84px;
            }

            .hero-title {
                font-size: 1.7rem;
            }

            .block-container {
                padding-top: 1.25rem;
            }
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-main: #081120;
                --bg-alt: #0b1426;
                --card-bg: rgba(15, 23, 42, 0.92);
                --text-main: #e5ecf6;
                --text-soft: #a9b7d0;
                --border-soft: #23314f;
                --axa-blue-soft: rgba(77, 113, 255, 0.16);
                --shadow-soft: 0 12px 28px rgba(0, 0, 0, 0.28);
                --shadow-strong: 0 18px 42px rgba(0, 0, 0, 0.34);
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(77, 113, 255, 0.10), transparent 28%),
                    linear-gradient(180deg, #081120 0%, #0d1728 100%);
                color: var(--text-main);
            }

            .panel-card,
            .metric-card,
            .tips-card {
                background: rgba(15, 23, 42, 0.92);
                border-color: var(--border-soft);
            }

            .panel-card.soft-blue,
            .metric-card,
            .tips-card {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.96) 0%, rgba(12, 19, 35, 0.96) 100%);
            }

            .section-title,
            .metric-value,
            .stTextArea label {
                color: #c8d7ff !important;
            }

            .stTextArea textarea {
                background: #0f172a !important;
                color: #e5ecf6 !important;
                border: 1px solid #2a3a5f !important;
            }

            .stTextArea textarea:focus {
                border: 1px solid #6d87ff !important;
                box-shadow: 0 0 0 3px rgba(109, 135, 255, 0.18) !important;
            }

            .badge {
                background: rgba(77, 113, 255, 0.14);
                color: #dbe5ff;
                border-color: #2a3a5f;
            }

            .tips-text {
                color: #ffffff;
            }

            .highlight-text {
                background: linear-gradient(transparent 58%, rgba(255, 255, 255, 0.20) 58%);
                color: #ffffff;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Clean text values by converting NaN to empty string and trimming whitespace
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


# Split comma-separated labels/tags into a normalized lowercase list
def split_tags(x):
    x = clean_text(x)
    if not x:
        return []
    return [i.strip().lower() for i in x.split(",") if i.strip()]


# Tokenize text into simple lowercase alphabetic tokens
def tokenize_simple(text):
    return re.findall(r"[a-zA-Z_]+", str(text).lower())


# Cache dataset loading to avoid re-reading files on every rerun
@st.cache_data
def load_data():
    # Check whether all required CSV files exist
    required_files = [MODEL1_PATH, MODEL2_PATH, SYMBOL_KB_PATH]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    # Load datasets
    df_model1 = pd.read_csv(MODEL1_PATH)
    df_model2 = pd.read_csv(MODEL2_PATH)
    df_symbol_kb = pd.read_csv(SYMBOL_KB_PATH)

    # Clean and normalize model1 dataset columns
    for col in ["dream_text", "stress_label", "emotion_labels", "theme_labels", "symbol_labels"]:
        df_model1[col] = df_model1[col].apply(clean_text)

    df_model1["stress_label"] = df_model1["stress_label"].str.lower().replace({"moderate": "medium"})
    df_model1 = df_model1[
        df_model1["stress_label"].isin({"low", "medium", "high"})
    ].drop_duplicates().reset_index(drop=True)

    # Convert label strings into list format for easier matching
    df_model1["emotion_list"] = df_model1["emotion_labels"].apply(split_tags)
    df_model1["theme_list"] = df_model1["theme_labels"].apply(split_tags)
    df_model1["symbol_list"] = df_model1["symbol_labels"].apply(split_tags)

    # Clean and normalize model2 dataset columns
    for col in ["stress_label", "emotion_labels", "dominant_emotion", "recommendation_text"]:
        df_model2[col] = df_model2[col].apply(clean_text)

    df_model2["stress_label"] = df_model2["stress_label"].str.lower().replace({"moderate": "medium"})
    df_model2 = df_model2[
        df_model2["stress_label"].isin({"low", "medium", "high", "very_high", "severe"})
    ].drop_duplicates().reset_index(drop=True)

    # Convert emotion labels into list format
    df_model2["emotion_list"] = df_model2["emotion_labels"].apply(split_tags)

    # Clean and normalize symbol knowledge base columns
    for col in ["symbol_name", "traditional_summary_en", "theme_tags", "emotion_hints", "stress_hint", "source_origin"]:
        df_symbol_kb[col] = df_symbol_kb[col].apply(clean_text)

    df_symbol_kb["symbol_name"] = df_symbol_kb["symbol_name"].str.lower()
    df_symbol_kb["theme_list"] = df_symbol_kb["theme_tags"].apply(split_tags)
    df_symbol_kb["emotion_list"] = df_symbol_kb["emotion_hints"].apply(split_tags)

    # Build dictionary for quick symbol lookup
    symbol_kb_dict = {row["symbol_name"]: row for _, row in df_symbol_kb.iterrows()}

    return df_model1, df_model2, df_symbol_kb, symbol_kb_dict


# Cache model loading so models are downloaded/loaded only once
@st.cache_resource
def load_models():
    # Load tokenizer and classifier for stress prediction
    stress_tokenizer = AutoTokenizer.from_pretrained(MODEL1_HF_NAME)
    stress_model = AutoModelForSequenceClassification.from_pretrained(MODEL1_HF_NAME).to(DEVICE)
    stress_model.eval()

    # Load tokenizer and seq2seq model for supportive text generation
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE)
    gen_model.eval()

    return stress_tokenizer, stress_model, gen_tokenizer, gen_model


# Load datasets and models
df_model1, df_model2, df_symbol_kb, symbol_kb_dict = load_data()
stress_tokenizer, stress_model, gen_tokenizer, gen_model = load_models()


# Predict stress level from dream text using the classifier model
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


# Find similar dream examples by word overlap
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


# Infer likely emotions from the most similar dream examples
def infer_emotions(text):
    matches = find_similar_examples(text, top_k=5)
    emotions = []

    for row in matches:
        emotions.extend(row["emotion_list"])

    emotions = pd.Series(emotions).value_counts().head(5).index.tolist() if emotions else []
    return emotions


# Map model1 stress labels to compatible model2 stress labels
def map_stress_for_model2(stress):
    if stress == "low":
        return ["low"]
    if stress == "medium":
        return ["medium"]
    if stress == "high":
        return ["high", "very_high"]
    return ["medium"]


# Retrieve the best matching recommendation from dataset 2
def retrieve_recommendation(pred_stress, inferred_emotions):
    stress_candidates = map_stress_for_model2(pred_stress)
    subset = df_model2[df_model2["stress_label"].isin(stress_candidates)].copy()

    # Fallback message if no matching rows are found
    if subset.empty:
        return "Take things one step at a time, reduce pressure, and focus on steady emotional recovery."

    emo_set = set(inferred_emotions)
    scored = []

    # Score recommendations by emotion overlap
    for _, row in subset.iterrows():
        overlap = len(emo_set.intersection(set(row["emotion_list"])))
        score = overlap + (1 if row["dominant_emotion"] in emo_set else 0)
        scored.append((score, row))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    return scored[0][1]["recommendation_text"]


# Build the prompt used for generating supportive well-being tips
def build_wellbeing_tips_prompt(dream_text, stress, emotions, fallback_tip):
    emotion_text = ", ".join(emotions[:5]) if emotions else "none clearly detected"

    return f"""
You are a supportive well-being assistant.

Dream text:
{dream_text}

Predicted stress level:
{stress}

Detected emotions:
{emotion_text}

Fallback support tip:
{fallback_tip}

Write well-being tips in 3 to 4 sentences.
Keep the tone gentle, practical, and supportive.
Focus on rest, reflection, emotional balance, and manageable next steps.
Do not mention diagnosis.
Do not mention mental illness.
""".strip()


# Generate text using the seq2seq model
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


# Detect poor-quality or invalid generated text
def is_bad_generated_text(text):
    if not text:
        return True

    lowered = text.lower().strip()

    bad_prefixes = [
        "dream text:",
        "predicted stress level:",
        "detected emotions:",
        "fallback support tip:",
        "you are a supportive",
        "write well-being tips",
    ]

    # Reject output if it looks like the prompt was repeated
    if any(lowered.startswith(prefix) for prefix in bad_prefixes):
        return True

    # Reject output if it is too short to be useful
    if len(text.split()) < 8:
        return True

    return False


# Main analysis pipeline for a dream
def analyze_dream(dream_text):
    # Step 1: Predict stress level
    stress, probs = predict_stress(dream_text)

    # Step 2: Infer emotions from similar examples
    emotions = infer_emotions(dream_text)

    # Step 3: Retrieve a fallback recommendation from dataset
    fallback_tip = retrieve_recommendation(stress, emotions)

    # Step 4: Generate well-being tips using the language model
    wellbeing_prompt = build_wellbeing_tips_prompt(
        dream_text=dream_text,
        stress=stress,
        emotions=emotions,
        fallback_tip=fallback_tip,
    )
    wellbeing_tips = generate_text(
        wellbeing_prompt,
        gen_tokenizer,
        gen_model,
        max_new_tokens=120,
    )

    # Step 5: If generated text looks invalid, use fallback tip instead
    if is_bad_generated_text(wellbeing_tips):
        wellbeing_tips = fallback_tip

    return {
        "dream_text": dream_text,
        "stress_level": stress,
        "stress_probs": probs,
        "emotions": emotions,
        "wellbeing_tips": wellbeing_tips,
    }


# Path to AXA logo image
logo_path = os.path.join(BASE_DIR, "axa_logo.png")

# Open main app wrapper
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

# Load and encode logo as base64 so it can be embedded directly in HTML
logo_html = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{encoded}" alt="AXA Health Logo" />'

# Render top hero/header section
st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="hero-grid">
            <div class="hero-logo-box">
                {logo_html}
            </div>
            <div>
                <div class="hero-title">AXA Dream Analyzer for Early Stress Awareness</div>
                <p class="hero-subtitle">
                    A wellness support tool that helps you reflect on dream themes and emotions, and offers stress awareness guidance.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render instructions panel
st.markdown(
    """
    <div class="panel-card soft-blue">
        <div class="section-title with-accent">How to use</div>
        <div class="guide-text">
            1. Type or paste your dream in the box below.<br>
            2. Click <b>Analyze Dream</b>.<br>
            3. Review the key themes, emotion cues and a stress level indicator.<br><br>
            <b>Important:</b> This tool provides general wellness support and self reflection only. It does not provide medical or mental health diagnosis, treatment, or emergency support. If you feel overwhelmed or unsafe, please seek professional help or local emergency services.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input text area for user dream description
dream_text = st.text_area(
    "Describe your dream",
    value="",
    height=200,
    placeholder="Example: I was running through a crowded station trying to find the right platform, but every sign kept changing..."
)

# Trigger analysis when button is clicked
if st.button("Analyze Dream"):
    if dream_text.strip():
        # Show loading spinner during analysis
        with st.spinner("Analyzing your dream..."):
            result = analyze_dream(dream_text.strip())

        st.success("Analysis completed successfully.")

        # Create two-column layout for result summary
        col1, col2 = st.columns(2)

        # Left column: stress level result
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Stress Level</div>
                    <div class="metric-value">{result["stress_level"].upper()}</div>
                    <div class="metric-helper">
                        This result reflects the overall stress pattern inferred from the dream narrative.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Right column: detected emotions result
        with col2:
            emotions_html = ""
            if result["emotions"]:
                emotions_html = '<div class="badge-row">' + "".join(
                    [f'<span class="badge">{emotion.title()}</span>' for emotion in result["emotions"]]
                ) + "</div>"
            else:
                emotions_html = '<div class="metric-helper">No clear emotions detected.</div>'

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Detected Emotions</div>
                    <div class="metric-value" style="font-size: 1.15rem;">Emotional Signals</div>
                    <div class="metric-helper">
                        The following emotional themes were most strongly associated with the narrative:
                    </div>
                    {emotions_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Display generated or fallback well-being tips
        st.markdown(
            f"""
            <div class="tips-card">
                <div class="section-title with-accent">Well-being Tips</div>
                <div class="tips-text">
                    <span class="highlight-text">{result["wellbeing_tips"]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Footer disclaimer
        st.markdown(
            """
            <div class="footer-note">
                This tool is intended for wellness support and reflective awareness only. It does not provide medical or clinical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Warn user if no dream text was entered
        st.warning("Please enter a dream before analysis.")

# Close main app wrapper
st.markdown('</div>', unsafe_allow_html=True)
