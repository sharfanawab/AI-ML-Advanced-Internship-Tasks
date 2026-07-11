import joblib

# Load model and vectorizer
model = joblib.load("models/news_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

labels = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

while True:
    text = input("\nEnter News: ")

    if text.lower() == "exit":
        break

    x = vectorizer.transform([text])

    prediction = model.predict(x)[0]

    print("Predicted Category:", labels[prediction])