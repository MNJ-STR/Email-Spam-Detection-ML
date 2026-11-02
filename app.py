
import streamlit as st
import joblib

st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="centered"
)

model = joblib.load("spam_model (1).pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("🛡️ SpamShield AI")

st.subheader("AI Based Spam Detection System")

st.write(
    "Detect spam messages using Machine Learning."
)

message = st.text_area(
    "Enter Your Message",
    placeholder="Paste your message here..."
)

if st.button("Check Spam", use_container_width=True):

    if message.strip():

        vector = tfidf.transform([message])

        prediction = model.predict(vector)

        probability = model.predict_proba(vector)

        spam_score = probability[0][1] * 100

        if prediction[0] == 1:
            st.error("🚨 SPAM MESSAGE DETECTED")
        else:
            st.success("✅ NOT SPAM MESSAGE")

        st.metric(
            "Spam Probability",
            f"{spam_score:.2f}%"
        )

        st.progress(float(spam_score / 100))

    else:
        st.warning("Please enter a message.")

st.divider()

st.caption("SpamShield AI | Developed using Python & ML")
