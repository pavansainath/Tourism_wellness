import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

HF_USERNAME = "pavansainath"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="best_model.joblib")
    return joblib.load(model_path)


model = load_model()

st.title("Wellness Tourism Package — Purchase Prediction")
st.write(
    "Enter a customer's profile and pitch-interaction details to predict whether "
    "they are likely to purchase the new Wellness Tourism Package."
)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 35)
    typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 1, 60, 15)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
    num_followups = st.number_input("Number of Follow-ups", 0, 10, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
with col2:
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_trips = st.number_input("Number of Trips per Year", 0, 20, 2)
    passport = st.selectbox("Holds Passport?", ["Yes", "No"])
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
    num_children = st.number_input("Number of Children Visiting", 0, 5, 0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", 1000, 100000, 20000)

input_df = pd.DataFrame([{
    "Age": age, "TypeofContact": typeofcontact, "CityTier": citytier,
    "DurationOfPitch": duration_of_pitch, "Occupation": occupation, "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting, "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched, "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status, "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0, "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0, "NumberOfChildrenVisiting": num_children,
    "Designation": designation, "MonthlyIncome": monthly_income,
}])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.success(f"This customer is **LIKELY** to purchase the Wellness Tourism Package (probability: {proba:.1%}).")
    else:
        st.warning(f"This customer is **UNLIKELY** to purchase the Wellness Tourism Package (probability: {proba:.1%}).")
