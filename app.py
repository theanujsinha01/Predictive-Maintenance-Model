import streamlit as st
import joblib
import pandas as pd
import numpy as np

# page title and page icon
st.set_page_config(
    page_title='Predictive Maintenance Model',
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
time_passed = st.number_input("Enter Time Passed (hours)", min_value=0, step=1)  # Time passed in hours

# Predict button
if st.button("Predict"):
    # Preprocess the input data
    input_data = pd.DataFrame({
        'ProductType': [product_type],
        'Humidity': [humidity],
        'Temperature': [temperature],
        'Age': [age],
    })

    try:
        
        transformed_data = preprocessor.transform(input_data)

        # Step 3: Make prediction
        prediction = model.predict(transformed_data)

        # Step 4: Display the result
        st.success(f"The predicted MTTF is {prediction[0]:.2f} hours")

        #-------------------------------------------------------------------------- 
        # Calculate reliability using the time passed (in hours)
        lambda_val = 1 / prediction[0]  # Failure rate (1/MTTF)

        # Calculate reliability based on time passed
        reliability = np.exp(-lambda_val * time_passed) * 100

        # Display reliability
        st.success(f"The reliability of the machine after {time_passed} hours is {reliability:.2f}%")

        #--------------------------------------------------------------------------

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


