import streamlit as st
import requests

st.title("📩 Real-Time Spam Detection")

message = st.text_area("Enter your message")

if st.button("Check"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"message": message}
        )

        result = response.json()

        if result["prediction"] == "spam":
            st.error(f"🚨 Spam (Confidence: {result['confidence']})")
        else:
            st.success(f"✅ Not Spam (Confidence: {result['confidence']})")