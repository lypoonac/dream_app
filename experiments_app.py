
import json
import os
import streamlit as st
import pandas as pd

from evaluate_models import run_model_selection
from evaluate_final import run_final_evaluation

st.set_page_config(page_title="Experiments Dashboard", page_icon="📊", layout="wide")

st.title("📊 Experiments Dashboard")

st.write("Run model selection and final application evaluation from Streamlit.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Selection")
    if st.button("Run Model Selection"):
        with st.spinner("Running model selection..."):
            try:
                results_df, metadata = run_model_selection()
                st.success("Model selection completed.")
                st.dataframe(results_df, use_container_width=True)

                st.markdown("### Metadata")
                st.json(metadata)
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    st.subheader("Final Application Evaluation")
    if st.button("Run Final Evaluation"):
        with st.spinner("Running final evaluation..."):
            try:
                results = run_final_evaluation()
                st.success("Final evaluation completed.")
                st.json(results)
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.subheader("Saved Result Files")

files_to_show = [
    "results/model_selection_results.csv",
    "results/model_selection_metadata.json",
    "results/final_app_results.json"
]

for file_path in files_to_show:
    if os.path.exists(file_path):
        st.write(f"✅ {file_path}")
    else:
        st.write(f"❌ {file_path} not found")

if os.path.exists("results/model_selection_results.csv"):
    st.markdown("### Latest Model Selection Table")
    df = pd.read_csv("results/model_selection_results.csv")
    st.dataframe(df, use_container_width=True)

if os.path.exists("results/final_app_results.json"):
    st.markdown("### Latest Final Evaluation Result")
    with open("results/final_app_results.json", "r") as f:
        data = json.load(f)
    st.json(data)
