import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.title("Email Spam Detection using SVM")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    X = data['text']
    y = data['label_num']

    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write("Accuracy: {:.2f}%".format(acc * 100))
    st.write("Precision: {:.2f}%".format(prec * 100))
    st.write("Recall: {:.2f}%".format(rec * 100))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center')

    st.pyplot(fig)

    # Custom Email Test
    st.subheader("Test Custom Email")
    user_input = st.text_area("Enter Email Text")

    if st.button("Predict"):
        if user_input.strip() != "":
            user_vector = vectorizer.transform([user_input])
            prediction = model.predict(user_vector)

            if prediction[0] == 1:
                st.error("SPAM Email")
            else:
                st.success("HAM Email")
        else:
            st.warning("Please enter some text")
