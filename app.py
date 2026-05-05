import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dummy model (for now)
model = LogisticRegression()
model.fit([[25, 50000], [50, 100000]], [0, 1])

st.title("Bank Customer Churn Predictor")

# Inputs
credit_score = st.number_input("Credit Score")
age = st.slider("Age", 18, 80)
balance = st.number_input("Balance")
tenure = st.slider("Tenure", 0, 10)
num_products = st.slider("Number of Products", 1, 4)
is_active = st.selectbox("Active Member", [0, 1])
salary = st.number_input("Estimated Salary")

geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Encoding
geo_france = 1 if geography == "France" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

# Prediction button
if st.button("Predict"):
    input_data = np.array([[credit_score, age, tenure, balance,
                            num_products, is_active, salary,
                            geo_france, geo_spain, gender_male]])

    prob = model.predict_proba(input_data)[0][1]

    if prob > 0.5:

        model = pickle.load(open("model.pkl", "rb"))
        st.error(f"⚠️ High Churn Risk: {prob:.2f}")
    else:
        st.success(f"✅ Low Churn Risk: {prob:.2f}")
