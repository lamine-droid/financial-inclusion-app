# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 19:52:06 2025

@author: THINKPAD
"""

import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')

st.title("Financial Inclusion Prediction")

country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
year = st.selectbox("Year", [2016, 2017, 2018])
location_type = st.selectbox("Location Type", ["Urban", "Rural"])
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
household_size = st.number_input("Household Size", min_value=1)
age = st.number_input("Age of Respondent", min_value=10, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])
relationship = st.selectbox("Relationship with Head", [
    "Head of Household", "Spouse", "Child", "Parent", 
    "Other relative", "Other non-relatives", "Dont know"
])
marital_status = st.selectbox("Marital Status", [
    "Married/Living together", "Divorced/Seperated", "Widowed", 
    "Single/Never Married", "Don’t know"
])
education_level = st.selectbox("Education Level", [
    "No formal education", "Primary education", "Secondary education", 
    "Vocational/Specialised training", "Tertiary education", "Other/Dont know/RTA"
])
job_type = st.selectbox("Job Type", [
    "Farming and Fishing", "Self employed", "Formally employed Government", 
    "Formally employed Private", "Informally employed", "Remittance Dependent", 
    "Government Dependent", "Other Income", "No Income", "Dont Know/Refuse to answer"
])

if st.button("Predict"):
    input_dict = {
        "country": [country],
        "year": [year],
        "location_type": [location_type],
        "cellphone_access": [cellphone_access],
        "household_size": [household_size],
        "age_of_respondent": [age],
        "gender_of_respondent": [gender],
        "relationship_with_head": [relationship],
        "marital_status": [marital_status],
        "education_level": [education_level],
        "job_type": [job_type]
    }

    input_df = pd.DataFrame(input_dict)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in input_df.columns:
        input_df[col] = le.fit_transform(input_df[col])

    prediction = model.predict(input_df)[0]
    result = "Has Bank Account" if prediction == 1 else "No Bank Account"
    st.success(f"Prediction: {result}")
    
    

    