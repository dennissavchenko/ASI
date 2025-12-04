import streamlit as st
import pandas as pd
import joblib

# Load trained model in joblib format
model = joblib.load("iris_model.joblib")

# Define the feature names manually since the model doesn’t have feature_names_in_
feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

# Define species names
species_names = ["iris-setosa", "iris-versicolor", "iris-virginica"]

# Streamlit UI
st.title("🪻Iris Species Predictor")
st.write("This app predicts the species of an iris based on its measurements.")

# Sidebar inputs
st.sidebar.header("Input Iris Features Manually")

def predict(features: list):
    df = pd.DataFrame(features)
    class_id = model.predict(df)
    return class_id

def probabilities(features: list):
    df = pd.DataFrame(features)
    proba = model.predict_proba(df)
    return pd.DataFrame(proba, columns=[f"{species_names[i]} ({proba[0][i] * 100:.1f}%)" for i in range(proba.shape[1])]).T

def user_input_features():
    inputs = {}
    for f in feature_names:
        val = st.sidebar.text_input(f, value="0.0", key=f"{f}_text")
        try:
            num = float(val)
            if num < 0:
                num = 0.0
        except ValueError:
            num = 0.0
        inputs[f] = num
    return [inputs]

# Get user input
input = user_input_features()

# Display input
st.subheader("🔍 Iris Features")
st.write(pd.DataFrame(input))

# Predict button
if st.button("Predict"):

    # Make prediction
    prediction = predict(input)
    st.subheader("🎯 Prediction Result")
    st.write(f"Predicted Specie: {species_names[prediction[0]]}")

    st.subheader("📊 Prediction Probabilities")
    st.bar_chart(probabilities(input))

