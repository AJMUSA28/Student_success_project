import streamlit as st
import pandas as pd
import joblib

# Load data
df = pd.read_csv(r"C:\Users\user\OneDrive\Documents\Data_Science_Freelancing_sample_projects\Student_success_project\data\student__data.csv")

# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Load model
model = joblib.load("models/student_model.pkl")

st.title("🎓 Student Success Analytics Dashboard")

# Dataset preview
st.subheader("Dataset Overview")
st.write(df.head())

# Target distribution
st.subheader("Target Distribution")
st.bar_chart(df['target'].value_counts())

# ----------------------------
# Prediction Section
# ----------------------------
st.subheader("Predict Student Outcome")

st.write("⚠️ NOTE: This model requires full feature input. This is a simplified demo.")

# Create FULL feature input using dataset structure
input_data = {}

for col in df.drop(['target'], axis=1).columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    
    prediction = model.predict(input_df)[0]

    # Decode output (optional if encoded)
    label_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    
    st.success(f"Prediction: {label_map.get(prediction, prediction)}")



    
    