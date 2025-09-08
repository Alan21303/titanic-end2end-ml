# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('model/model.pkl')

# Custom page config
st.set_page_config(page_title="Titanic Survival Predictor 🚢", layout="centered")

# Title
st.title("🚢 Titanic Survival Prediction App")
st.markdown("### Enter passenger details below to predict survival:")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About Features")
    st.markdown("""
    - **Passenger Class**: First > Second > Third class survival.
    - **Sex**: Women had much higher survival (women & children first).
    - **Age**: Children more likely to survive.
    - **Siblings/Spouses (SibSp)**: Having 1–2 helped, too many reduced chances.
    - **Parents/Children (Parch)**: Moderate family = better odds.
    - **Fare**: Higher fare = wealthier class = better survival.
    - **Port of Embarkation**:
        - **Cherbourg (C)** → higher survival  
        - **Southampton (S)** → lower survival  
        - **Queenstown (Q)** → lowest survival
    """)

# Input container
with st.container():
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

# Convert values back for model
pclass_map = {"First Class": 1, "Second Class": 2, "Third Class": 3}
sex_map = {"Male": "male", "Female": "female"}
embarked_map = {"Cherbourg": "C", "Queenstown": "Q", "Southampton": "S"}

input_data = pd.DataFrame([{
    'Pclass': pclass_map[pclass],
    'Sex': sex_map[sex],
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked_map[embarked]
}])

# Buttons row
colA, colB = st.columns([1,1])

with colA:
    if st.button("🔮 Predict Survival"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"🎉 The passenger is likely to **SURVIVE** (Probability: {probability:.2f})")
        else:
            st.error(f"☠️ The passenger is likely **NOT to survive** (Probability: {probability:.2f})")

with colB:
    if st.button("🎯 Try a Survivor Case"):
        st.info("Auto-filled with a likely survivor profile!")
        st.write("""
        **Example Survivor**  
        - Passenger Class: First Class  
        - Sex: Female  
        - Age: 8  
        - Siblings/Spouses: 1  
        - Parents/Children: 2  
        - Fare: 100  
        - Port of Embarkation: Cherbourg  
        """)

