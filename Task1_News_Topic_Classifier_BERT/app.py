import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load(
    "models/news_classifier.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)
# Categories
categories = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Technology"
}
# App Title
st.title(" News Topic Classifier")

st.write(
    "Enter a news headline and the model will predict its category."
)
# User Input
news_text = st.text_area(
    "Enter News Headline"
)
# Button
if st.button("Predict"):
    if news_text:
        # Convert text
        text_vector = vectorizer.transform(
            [news_text]
        )
        # Prediction
        prediction = model.predict(
            text_vector
        )
        result = categories[prediction[0]]
        st.success(
            f"Predicted Topic: {result}"
        )
    else:
        st.warning(
            "Please enter a news headline"
        )