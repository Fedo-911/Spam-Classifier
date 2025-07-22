import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📧 Email Spam Classifier")

# User input
email = st.text_area("Enter your email message:")

if st.button("Check"):
    if email.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed = vectorizer.transform([email])
        prediction = model.predict(transformed)
        if prediction[0] == 1:
            st.error("🚫 Spam Email")
        else:
            st.success("✅ Not Spam")
