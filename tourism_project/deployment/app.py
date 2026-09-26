import gradio as gr
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

HF_USERNAME = "pavansainath"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"

model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="best_model.joblib")
model = joblib.load(model_path)

def predict(age, typeofcontact, citytier, duration_of_pitch, occupation, gender,
            num_person_visiting, num_followups, product_pitched, preferred_star,
            marital_status, num_trips, passport, pitch_score, own_car,
            num_children, designation, monthly_income):
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
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    label = "LIKELY to purchase" if pred == 1 else "UNLIKELY to purchase"
    return f"{label} the Wellness Tourism Package (probability: {proba:.1%})"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age", value=35, minimum=18, maximum=100),
        gr.Radio(["Self Enquiry", "Company Invited"], label="Type of Contact", value="Self Enquiry"),
        gr.Radio([1, 2, 3], label="City Tier", value=1),
        gr.Number(label="Duration of Pitch (minutes)", value=15, minimum=1, maximum=60),
        gr.Dropdown(["Salaried", "Small Business", "Large Business", "Free Lancer"], label="Occupation", value="Salaried"),
        gr.Radio(["Male", "Female"], label="Gender", value="Male"),
        gr.Number(label="Number of Persons Visiting", value=2, minimum=1, maximum=10),
        gr.Number(label="Number of Follow-ups", value=3, minimum=0, maximum=10),
        gr.Dropdown(["Basic", "Deluxe", "Standard", "Super Deluxe", "King"], label="Product Pitched", value="Basic"),
        gr.Radio([3.0, 4.0, 5.0], label="Preferred Property Star", value=3.0),
        gr.Dropdown(["Single", "Married", "Divorced", "Unmarried"], label="Marital Status", value="Single"),
        gr.Number(label="Number of Trips per Year", value=2, minimum=0, maximum=20),
        gr.Radio(["Yes", "No"], label="Holds Passport?", value="No"),
        gr.Slider(1, 5, value=3, step=1, label="Pitch Satisfaction Score"),
        gr.Radio(["Yes", "No"], label="Owns a Car?", value="Yes"),
        gr.Number(label="Number of Children Visiting", value=0, minimum=0, maximum=5),
        gr.Dropdown(["Executive", "Manager", "Senior Manager", "AVP", "VP"], label="Designation", value="Executive"),
        gr.Number(label="Monthly Income", value=20000, minimum=1000, maximum=100000),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Wellness Tourism Package — Purchase Prediction",
    description="Enter a customer's profile and pitch-interaction details to predict whether they are likely to purchase the new Wellness Tourism Package.",
)

if __name__ == "__main__":
    demo.launch()
