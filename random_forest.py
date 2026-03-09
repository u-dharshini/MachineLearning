import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Titanic Survival Prediction using Random Forest")

# Upload dataset
uploaded_file = st.file_uploader("Upload Titanic train.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Select useful columns
    data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

    # Handle missing values
    data['Age'].fillna(data['Age'].mean(), inplace=True)

    # Convert gender to numeric
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # Split features and target
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {acc*100:.2f}%")

    st.subheader("Enter Passenger Details")

    # UI Inputs
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
    sibsp = st.number_input("Siblings/Spouse", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=50.0)

    if st.button("Predict Survival"):

        sex_value = 0 if sex == "male" else 1

        new_passenger = [[pclass, sex_value, age, sibsp, parch, fare]]

        prediction = model.predict(new_passenger)
        prob = model.predict_proba(new_passenger)[0]

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("PASSENGER SURVIVED")
        else:
            st.error("PASSENGER DID NOT SURVIVE")

        st.write("Prediction Probabilities:")
        st.write(f"Did Not Survive: {prob[0]*100:.2f}%")
        st.write(f"Survived: {prob[1]*100:.2f}%")

        # Graph
        labels = ["Did Not Survive", "Survived"]
        values = [prob[0], prob[1]]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Probability")
        ax.set_title("Survival Prediction Probability")

        st.pyplot(fig)