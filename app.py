import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os


# page tittle and page icon
st.set_page_config(
    page_title= 'Predictive Maintenance Model',
    page_icon='🔧'
)

# Load the saved model and preprocessor
model = joblib.load('predictive_maintenance_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')


   


# Define the app interface
st.title("Predictive Maintenance Model")
st.markdown("### Predict the Mean Time to Failure (MTTF) for industrial equipment")

# Create input fields for user data
product_type = st.selectbox("Select Product Type", ['Gauge Machine', 'Extruder', 'Pump', 'Coil Oven', 'Pressure Cutter'])
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
temperature = st.number_input("Enter Temperature (°C)", min_value=-50.0, max_value=150.0, step=0.1)
age = st.number_input("Enter Equipment Age (years)", min_value=0, max_value=100, step=1)
quantity = st.number_input("Enter Quantity (units)", min_value=0, max_value=10000, step=1)

# Predict button
if st.button("Predict"):
    # Preprocess the input data
    input_data = pd.DataFrame({
        'ProductType': [product_type],
        'Humidity': [humidity],
        'Temperature': [temperature],
        'Age': [age],
        'Quantity': [quantity]
    })
    
    try:
        # Transform the input data
        transformed_data = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(transformed_data)
        
        # Display result
        st.success(f"The predicted MTTF is {prediction[0]:.2f} hours")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
