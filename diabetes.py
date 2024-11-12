# Import necessary packages
import streamlit as st
import pickle  # Use pickle for loading the model
import numpy as np
from PIL import Image

# Information about the model's input attributes
attribute = ("""
## Attribute Information

- Pregnancies: The number of pregnancies you had.
- Glucose: Glucose level in your blood in mg/dL
- Blood Pressure: Blood Pressure value mmHg
- Skin Thickness: Skin Thickness value in mm
- Insulin: Insulin level in blood mmol/L 
- BMI: Body Mass Index
- Diabetes Pedigree Function: Diabetes Pedigree Value
- Age: Age in years
""")

# Function to run the diabetes prediction model
def diabetes_pred():
    st.write("# Diabetes Predictor")
    img = Image.open("diabetes.jpg")
    st.image(img)
    
    st.write("""
    The Diabetes Predictor App is a powerful tool designed to help individuals assess their risk of developing diabetes. 
    It uses machine learning algorithms to analyze risk factors like age, BMI, blood pressure, and glucose levels to estimate 
    diabetes risk. This app can empower individuals to take preventive action or adjust lifestyle choices to maintain better health.
    """)
    
    st.markdown(attribute)
    
    # Get user input
    st.header("Give Your Input")
    
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Enter the number of Pregnancies you had", 0, 50, 2)
        blood_pressure = st.number_input("Enter your Blood Pressure", 0, 360, 80)
        insulin = st.number_input("Enter your Insulin Level", 0, 1000, 100)
        diabetes_pedigree_function = st.number_input("Enter your Diabetes Pedigree Function Value", 0.0, 5.0, 0.05, step=0.001)
        
    with col2:
        glucose = st.number_input("Enter your Glucose Level", 0, 210, 130)
        skin_thickness = st.number_input("Enter the Thickness of your skin", 0, 200, 20)
        bmi = st.number_input("Enter your BMI value", 15.0, 100.0, 25.0, step=0.1)
        age = st.number_input("Enter your Age", 12, 100, 20)
    
    # Show the selected options
    with st.expander("Your selected options"):
        user_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree_function,
            "Age": age
        }
        st.write(user_data)
        
        # Convert user input into a list format
        input_data = [float(value) for value in user_data.values()]

    # Prediction results
    with st.expander("Prediction Results"):
        input_array = np.array(input_data).reshape(1, -1)
        
        # Load the model with pickle
        try:
            with open("diabetes_model.pkl", "rb") as file:
                model = pickle.load(file)
                
            # Perform prediction
            prediction = model.predict(input_array)
            probability = model.predict_proba(input_array)
            
            # Display the results
            if prediction == 1:
                st.warning("Positive Risk! You may have Diabetes. Please consult a healthcare professional.")
                prob_score = {
                    "Positive Risk": probability[0][1],
                    "Negative Risk": probability[0][0]
                }
                st.write(prob_score)
            else:
                st.success("Negative Risk! You do not have Diabetes. Keep up the healthy habits!")
                prob_score = {
                    "Negative Risk": probability[0][0],
                    "Positive Risk": probability[0][1]
                }
                st.write(prob_score)
        
        except FileNotFoundError:
            st.error("Model file not found. Please check that 'diabetes_model.pkl' is in the correct directory.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
