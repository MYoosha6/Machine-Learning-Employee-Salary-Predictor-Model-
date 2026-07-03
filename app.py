import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'best_model.pkl'
DATA_PATH = BASE_DIR / 'data' / 'job_salary_prediction_dataset.csv'

# Configure the app
st.set_page_config(page_title="AI-Based Employee Salary Prediction System", layout="wide")
st.title("AI-Based Employee Salary Prediction System")
st.write("Enter the employee details below to predict their expected salary.")

# Cache loading data to extract unique categorical values for selectboxes
@st.cache_data
def load_categorical_options():
    try:
        df = pd.read_csv(DATA_PATH)
        options = {
            'job_title': sorted(df['job_title'].unique().tolist()),
            'education_level': sorted(df['education_level'].unique().tolist()),
            'industry': sorted(df['industry'].unique().tolist()),
            'company_size': sorted(df['company_size'].unique().tolist()),
            'location': sorted(df['location'].unique().tolist()),
            'remote_work': sorted(df['remote_work'].unique().tolist())
        }
        return options
    except Exception as e:
        st.error(f"Failed to load dataset for dropdown options: {e}")
        return None

# Cache model loading
@st.cache_resource
def load_model():
    try:
        # best_model.pkl is the full Scikit-Learn Pipeline containing both the
        # ColumnTransformer (preprocessor) and the Regressor (RandomForest).
        # This satisfies the requirement to load both the model and preprocessing pipeline.
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load the trained model/preprocessor: {e}")
        return None

options = load_categorical_options()
model = load_model()

if options is None or model is None:
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox("Job Title", options['job_title'])
    experience_years = st.number_input("Experience Years", min_value=0, max_value=60, value=5, step=1)
    education_level = st.selectbox("Education Level", options['education_level'])
    skills_count = st.number_input("Skills Count", min_value=0, max_value=100, value=5, step=1)
    industry = st.selectbox("Industry", options['industry'])

with col2:
    company_size = st.selectbox("Company Size", options['company_size'])
    location = st.selectbox("Location", options['location'])
    remote_work = st.selectbox("Remote Work", options['remote_work'])
    certifications = st.number_input("Certifications", min_value=0, max_value=50, value=0, step=1)

# Prediction Button
if st.button("Predict Salary"):
    # Validate numerical inputs (extra safety layer on top of Streamlit's min_value)
    if experience_years < 0 or skills_count < 0 or certifications < 0:
        st.warning("Experience, Skills Count, and Certifications must be non-negative.")
    else:
        # Create input dataframe matching the original schema
        input_data = pd.DataFrame([{
            'job_title': job_title,
            'experience_years': experience_years,
            'education_level': education_level,
            'skills_count': skills_count,
            'industry': industry,
            'company_size': company_size,
            'location': location,
            'remote_work': remote_work,
            'certifications': certifications
        }])

        try:
            # Predict
            predicted_salary = model.predict(input_data)[0]
            
            # Display prediction cleanly
            st.success(f"### Predicted Salary: ${predicted_salary:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
