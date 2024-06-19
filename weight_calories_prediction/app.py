import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load the model
model = load_model(r'model_new.h5')
# Load the scaler

scaler = joblib.load(r'scaler.joblib')

# Function to calculate BMR (Basal Metabolic Rate)
def calculate_bmr(gender, age, body_height, body_weight, activity):
    body_height_cm = body_height * 100
    if gender == 0:  # Female
        bmr = 447.593 + (9.247 * body_weight) + (3.098 * body_height_cm) - (4.330 * age)
    else:  # Male
        bmr = 88.362 + (13.397 * body_weight) + (4.799 * body_height_cm) - (5.677 * age)

    if activity == 1:
        tdee = bmr * 1.465
    elif activity == 2:
        tdee = bmr * 1.2
    else:
        tdee = bmr * 1.8125

    return tdee

# Obesity level mapping
obesity_mapping = {
    0: 'Insufficient Weight',
    1: 'Normal Weight',
    2: 'Obesity Type I',
    3: 'Obesity Type II',
    4: 'Obesity Type III',
    5: 'Overweight Level I',
    6: 'Overweight Level II'
}

def classify(gender, age, body_height, body_weight, activity_level):
    calories = calculate_bmr(gender, age, body_height, body_weight, activity_level)

    # Prepare the input data for the model
    user_data = np.array([[gender, age, body_height, body_weight, activity_level]])
    user_data = scaler.transform(user_data)
    predictions = model.predict(user_data)
    predicted_weight_status_index = np.argmax(predictions)
    predicted_weight_status = int(predicted_weight_status_index)


    result = {
        "predicted_calories": int(calories),
        # "predicted_obesity": int(predicted_weight_status),
        "obesity_label": obesity_mapping[int(predicted_weight_status)],
        # "user_data": user_data,
        # "loaded_accuracy": loaded_accuracy
    }

    return result

def main():
    st.title("Obesity and Calorie Prediction")

    # Input fields
    gender = st.radio("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    body_height = st.number_input("Height (in meters)", min_value=0.5, max_value=2.5, value=1.75)
    body_weight = st.number_input("Weight (in kg)", min_value=0.0, max_value=300.0, value=70.0)
    activity_level = st.selectbox("Activity Level", options=[1, 2, 3], format_func=lambda x: "Low" if x == 1 else ("Medium" if x == 2 else "High"))

    if st.button("Predict"):
        result = classify(gender, age, body_height, body_weight, activity_level)
        st.write(f"Predicted Calories: {result['predicted_calories']}")
        # st.write(f"Predicted Obesity Level: {result['predicted_obesity']}")
        st.write(f"Obesity Label: {result['obesity_label']}")
        # st.write(f"loaded_accuracy: {result['loaded_accuracy']}")

if __name__ == '__main__':
    main()
