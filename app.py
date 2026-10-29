import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Streamlit page
st.set_page_config(
    page_title="Amazon Reviews Sentiment Analysis",
    page_icon="🛒"
)

st.title("🛒 Amazon Product Reviews Analytics")

st.write(
    "Enter an Amazon product review below to predict its sentiment."
)

# User input
review = st.text_area(
    "Enter your review:",
    placeholder="Example: This product is very good and useful."
)

# Prediction button
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        # Preprocess review
        cleaned_review = preprocess_text(review)

        # Convert text into TF-IDF
        review_vector = vectorizer.transform([cleaned_review])

        # Prediction
        prediction = model.predict(review_vector)[0]

        # Display result
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("😊 Positive Review")
        elif prediction == 0:
            st.error("😞 Negative Review")
        else:
            st.info(f"Predicted Sentiment: {prediction}")

