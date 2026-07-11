import pandas as pd

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

train_df["text"] = train_df["Title"] + " " + train_df["Description"]
test_df["text"] = test_df["Title"] + " " + test_df["Description"]

train_df.rename(columns={"Class Index":"label"}, inplace=True)
test_df.rename(columns={"Class Index":"label"}, inplace=True)

train_df["label"] = train_df["label"] - 1
test_df["label"] = test_df["label"] - 1

print(train_df.head())


from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"],
    train_df["label"],
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(train_texts))
print("Validation Samples:", len(val_texts))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

print("TF-IDF Train Shape:", X_train.shape)
print("TF-IDF Validation Shape:", X_val.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

model = LogisticRegression(max_iter=1000)

model.fit(X_train, train_labels)

predictions = model.predict(X_val)

accuracy = accuracy_score(val_labels, predictions)

print("Validation Accuracy:", accuracy)

f1 = f1_score(
    val_labels,
    predictions,
    average="weighted"
)

print("F1 Score:", f1)


report = classification_report(
    val_labels,
    predictions
)

print(report)

import os

os.makedirs("results", exist_ok=True)

with open("results/evaluation_report.txt", "w") as file:
    file.write(report)
    file.write("\n\nAccuracy: ")
    file.write(str(accuracy))
    file.write("\nF1 Score: ")
    file.write(str(f1))

print("Evaluation Report Saved!")

import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))
plt.bar(["Validation Accuracy"], [accuracy])
plt.ylim(0,1)
plt.title("Validation Accuracy")
plt.savefig("images/accuracy.png")
plt.close()

print("Accuracy graph saved!")

import joblib

joblib.dump(model, "models/news_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model Saved Successfully!")


import os
import joblib

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "models/news_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print(" Model Saved Successfully!")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix

cm = confusion_matrix(
    val_labels,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

plt.figure(figsize=(6,6))

disp.plot()

plt.title("News Classification Confusion Matrix")

plt.savefig(
    "images/confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

print("Confusion Matrix saved!")