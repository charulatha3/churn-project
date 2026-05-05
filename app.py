import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit([[25, 50000], [50, 100000]], [0, 1])

st.title("Bank Customer Churn Predictor")

age = st.slider("Age", 18, 80)
balance = st.number_input("Balance")

if st.button("Predict"):
    input_data = np.array([[age, balance]])
    prediction = model.predict_proba(input_data)[0][1]
    st.write(f"Churn Probability: {prediction:.2f}")
