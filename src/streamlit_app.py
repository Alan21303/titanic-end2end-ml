# streamlit_app.py
"""
Titanic Survival Prediction App (Streamlit)

A simple web interface to predict survival probability
of Titanic passengers using a trained ML model.
"""

import joblib
import pandas as pd
import streamlit as st

# Load trained model
model = joblib.load("model/model.pkl")

st.set_page_config(page_title="Titanic Survival Predictor 🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.markdown("### Enter passenger details below to predict survival:")

with st.sidebar:
    st.header("ℹ️ About Features")
    st.markdown("""
    - **Pclass**: 1st > 2nd > 3rd survival chances  
    - **Sex**: Women had higher survival odds  
    - **Age**: Younger = more likely to survive  
    - **SibSp**: Small families helped, large reduced chances  
    - **Parch**: Moderate family improved survival  
    - **Fare**: Higher fare = higher class → better odds  
    - **Embarked**: C > S > Q in survival
    """)

st.subheader("🎫 Passenger Details")
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", ["First Class", "Second Class", "Third Class"])
    sex = st.radio("Sex", ["Male", "Female"])
    age = st.slider("Age", 0, 80, 25)

with col2:
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
    fare = st.slider("Fare (Ticket Price)", 0, 500, 50)
    embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

# Map input values
pclass_map = {"First Class": 1, "Second Class": 2, "Third Class": 3}
sex_map = {"Male": "male", "Female": "female"}
embarked_map = {"Cherbourg": "C", "Queenstown": "Q", "Southampton": "S"}

input_data = pd.DataFrame([{
    "Pclass": pclass_map[pclass],
    "Sex": sex_map[sex],
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked_map[embarked],
}])

colA, colB = st.columns(2)

with colA:
    if st.button("🔮 Predict Survival"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"🎉 Passenger likely to **SURVIVE** (Probability: {probability:.2f})")
        else:
            st.error(f"☠️ Passenger likely **NOT to survive** (Probability: {probability:.2f})")

with colB:
    if st.button("🎯 Try a Survivor Case"):
        st.info("Example profile of a likely survivor:")
        st.write("""
        - Pclass: First Class  
        - Sex: Female  
        - Age: 8  
        - SibSp: 1  
        - Parch: 2  
        - Fare: 100  
        - Embarked: Cherbourg  
        """)