# AXA Health Dream Analyzer

AXA Health Dream Analyzer is a Streamlit-based wellness support application that analyzes a user's dream description to estimate a **stress level**, infer likely **emotional signals**, and generate gentle **well-being tips**.

The application combines:
1. a fine-tuned Hugging Face text classification model for stress prediction, and
2. a sequence-to-sequence text generation model for supportive recommendation generation.

> **Disclaimer:** This tool is intended for wellness support and reflective awareness only. It does **not** provide medical, psychological, or psychiatric diagnosis, treatment, or emergency support.

---

## Features

- Dream text input through a Streamlit web interface
- Predicted stress level:
  - Low
  - Medium
  - High
- Emotion inference using similar examples from a local dataset
- Supportive well-being tips generated with a Hugging Face text generation model
- Cached dataset and model loading for faster repeated use
- Custom AXA-themed UI styling

---

## Project Structure

```text
project/
├── app.py
├── axa_logo.png
├── data/
│   ├── model1_train.csv
│   ├── model2_train.csv
│   └── symbol_kb.csv
├── requirements.txt
└── README.md
