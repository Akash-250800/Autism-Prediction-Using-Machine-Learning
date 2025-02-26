import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model, scaler, and encoders
@st.cache_resource
def load_artifacts():
    # Load the model
    with open('autism_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load the scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load the dataset to infer structure
    data = pd.read_csv('train.csv')
    feature_cols = [col for col in data.columns if col != 'austim']  # Adjust 'diagnosis' to your target column
    
    # Load label encoders for categorical columns
    categorical_cols = data[feature_cols].select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        with open(f'{col}_label_encoder.pkl', 'rb') as f:
            label_encoders[col] = pickle.load(f)
    
    # Load target encoder if it exists
    try:
        with open('target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
    except FileNotFoundError:
        target_encoder = None
    
    numerical_cols = data[feature_cols].select_dtypes(include=['int64', 'float64']).columns
    
    return model, scaler, label_encoders, target_encoder, feature_cols, categorical_cols, numerical_cols

model, scaler, label_encoders, target_encoder, feature_cols, categorical_cols, numerical_cols = load_artifacts()

# Prediction function
def predict_autism(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])
    
    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Make prediction
    prediction = model.predict(input_df[feature_cols])
    
    # Decode prediction if target encoder exists
    if target_encoder:
        prediction = target_encoder.inverse_transform(prediction)
    
    return prediction[0]

# Streamlit app
st.title("Autism Prediction App")
st.write("Enter the details below to predict autism likelihood based on a pre-trained model.")

# Create input form dynamically based on features
input_data = {}
with st.form(key='prediction_form'):
    for col in feature_cols:
        if col in categorical_cols:
            # Get unique categories from the encoder's classes
            options = list(label_encoders[col].classes_)
            input_data[col] = st.selectbox(f"{col}", options)
        else:
            # Assume numerical inputs
            input_data[col] = st.number_input(f"{col}", step=0.1, format="%.2f")
    
    submit_button = st.form_submit_button(label="Predict")

# Handle prediction
if submit_button:
    try:
        prediction = predict_autism(input_data)
        st.success("Prediction completed!")
        st.write(f"Predicted Outcome: **{prediction}**")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")