import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide"
)

# App title and description
st.title("Loan Approval Prediction System")
st.markdown("""
This application predicts whether a loan application will be approved based on
applicant information. Fill in the form below to get a prediction.
""")

# Load the saved model
@st.cache_resource
def load_model():
    try:
        with open('loan_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'loan_model.pkl' is in the same directory as this app.")
        return None

model = load_model()

# Create input form
st.header("Applicant Information")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Married", "Not Married"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands $)", min_value=0, value=100)
    loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 60, 36, 12])
    credit_history = st.selectbox("Credit History", ["Good", "Poor"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Function to preprocess input data
def preprocess_input():
    # Create a dictionary of features
    data = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Married" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 1 if education == "Graduate" else 0,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': 1 if credit_history == "Good" else 0,
        'Property_Area_Rural': 1 if property_area == "Rural" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0
    }
    
    # Create DataFrame with the exact column order expected by your model
    # Note: You might need to adjust these columns based on your actual model
    features = pd.DataFrame(data, index=[0])
    return features

# Prediction button
if st.button("Predict Loan Approval"):
    if model is not None:
        # Preprocess input
        input_data = preprocess_input()
        
        # Make prediction
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            # Display prediction
            st.header("Prediction Result")
            
            if prediction[0] == 1:
                st.success("✅ Loan Approved!")
                st.balloons()
            else:
                st.error("❌ Loan Rejected")
            
            # Display confidence
            st.subheader("Prediction Confidence")
            approval_confidence = prediction_proba[0][1] * 100
            st.progress(int(approval_confidence))
            st.write(f"Confidence: {approval_confidence:.2f}%")
            
            # Display factors that might have influenced the decision
            st.subheader("Important Factors")
            factors = []
            
            if credit_history == "Poor":
                factors.append("❌ Poor credit history significantly reduces approval chances")
            else:
                factors.append("✅ Good credit history increases approval chances")
                
            if applicant_income < 3000:
                factors.append("❌ Low applicant income may reduce approval chances")
            elif applicant_income > 10000:
                factors.append("✅ High applicant income increases approval chances")
                
            if loan_amount > 200:
                factors.append("❌ High loan amount may reduce approval chances")
            
            for factor in factors:
                st.write(factor)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("Please check that your input data matches what the model expects.")

# Expander for model information
with st.expander("About this Model"):
    st.write("""
    This application uses a machine learning model trained on historical loan approval data.
    The model considers various factors like income, credit history, and loan amount to predict
    whether a loan will be approved.
    
    **Note:** This is a predictive tool and the actual loan approval decision may depend on
    additional factors not considered by this model.
    """)

# Sidebar for additional options
st.sidebar.header("Options")
if st.sidebar.checkbox("Show input data summary"):
    st.sidebar.subheader("Input Data Summary")
    input_data = preprocess_input()
    st.sidebar.write(input_data)

# Footer
st.markdown("""
---
Loan Prediction System | Created by Ojo Gbenga Charles
""")
