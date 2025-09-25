import streamlit as st
import requests
import joblib
from pathlib import Path

api_url = "https://python-ml-applications.onrender.com/"

pkl_path = Path(__file__).parent / "rf_model_pca.pkl"

with open(pkl_path, 'rb') as f:
    saved = joblib.load(f)
model_columns = saved["columns"]

st.title("Crocodile Prediction")

observed_length = st.number_input("Observed Length (m)")
observed_weight = st.number_input("Observed Weight (kg)")

scientific_name_options = [col.replace("Scientific Name_", "") for col in model_columns if col.startswith("Scientific Name_")]
scientific_name = st.selectbox("Scientific Name", scientific_name_options)

genus_options = [col.replace("Genus_", "") for col in model_columns if col.startswith("Genus_")]
genus = st.selectbox("Genus", genus_options)

age_class_options = [col.replace("Age Class_", "") for col in model_columns if col.startswith("Age Class_")]
age_class = st.selectbox("Age Class", age_class_options)

sex_options = [col.replace("Sex_", "") for col in model_columns if col.startswith("Sex_")]
sex = st.selectbox("Sex", sex_options)

habitat_options = [col.replace("Habitat Simple_", "") for col in model_columns if col.startswith("Habitat Simple_")]
habitat_simple = st.selectbox("Habitat", habitat_options)

if st.button("Predict"):
    data = {
        "observed_length": observed_length,
        "observed_weight": observed_weight,
        "scientific_name": scientific_name,
        "genus": genus,
        "age_class": age_class,
        "sex": sex,
        "habitat_simple": habitat_simple
    }
    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Prediction: {prediction}")
    else:
        st.error("API request failed")
