import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained model
with open("etc.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the pre-trained scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Title of the app
st.title(":red[social media prediction]")

# Description of the app
st.write("This is a social media prediction app.")

# User inputs for prediction
months_as_customer = st.number_input("Months as Customer", min_value=0, max_value=100, value=25)
capital_gains = st.number_input("Capital Gains", min_value=0, max_value=100000, value=1000)
capital_loss = st.number_input("Capital Loss", min_value=0, max_value=100000, value=100)
incident_hour = st.slider("Incident Hour of the Day", min_value=0, max_value=23, value=12)
bodily_injuries = st.radio("Bodily Injuries", [0, 1, 2])
injury_claim = st.number_input("Injury Claim", min_value=0, max_value=100000, value=5000)
property_claim = st.number_input("Property Claim", min_value=0, max_value=100000, value=3000)
vehicle_claim = st.number_input("Vehicle Claim", min_value=0, max_value=100000, value=7000)

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'months_as_customer': [months_as_customer],
    'capital-gains': [capital_gains],
    'capital-loss': [capital_loss],
    'incident_hour_of_the_day': [incident_hour],
    'injury_claim': [injury_claim],
    'property_claim': [property_claim],
    'vehicle_claim': [vehicle_claim],
    'bodily_injuries': [bodily_injuries]
})

                         # Apply scaling to the numeric features
input_data= scaler.transform(input_data)







# Prediction button
if st.button("Make Prediction"):
    # Make prediction
    try:
        prediction = model.predict(input_data)
        prediction = prediction.tolist()

        # Display the prediction result
        st.write("Prediction:", prediction[0])
        st.balloons()
    except Exception as e:
        st.error(f"Error in making prediction: {e}")
