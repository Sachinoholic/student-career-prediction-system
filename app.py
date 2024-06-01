import streamlit as st
import pickle
import numpy as np
import os
import joblib
import pandas as pd


@st.cache_resource()
def load_model():
    model_path = joblib.load("new_tree.pkl")
    return model_path

@st.cache_resource()
def load_preprocessor():
    pre=joblib.load("new_encoder.pkl")
    return pre




st.title('Student Career Predication (DATA SCIENCE)')

st.write("Please provide the following information:")

good_coder = st.selectbox('do you like coding/ a good coder?', ['yes', 'no'])
intrest_in_ml = st.selectbox('Are you intrested in ML?', ['yes', 'no'])
intrest_in_research = st.selectbox('Are you interested in research?', ['yes', 'no'])
intrest_in_ml_model_for_ds = st.selectbox('intrested in using ML models to solve DS problems?', ['yes', 'no'])
intrest_to_build_data_infrastructure = st.selectbox('intrested in build and manage data infrastructure?', ['yes', 'no'])
enjoy_consulting = st.selectbox('do you enjoy consulting and advising AI solutions?', ['yes', 'no'])
intrest_in_business_insights = st.selectbox('intrested in business insights and decision making?', ['yes', 'no'])

user_input = {
        'good_coder':[good_coder],
        'intrest_in_ml':[intrest_in_ml],
        'intrest_in_research':[intrest_in_research],
        'intrest_in_ml_model_for_ds':[intrest_in_ml_model_for_ds],
        'intrest_to_build_data_infrastructure':[intrest_to_build_data_infrastructure],
        'enjoy_consulting':[enjoy_consulting],
        'intrest_in_business_insights':[intrest_in_business_insights]}

data=pd.DataFrame(user_input)

if st.button('Predict'):
    
    preprocessor=load_preprocessor()
    main_model=load_model()
    
    encoded_array = preprocessor.transform(data)

    prediction = main_model.predict(encoded_array)
    
    st.write(f"The Suggested career is: {prediction[0]}")
   