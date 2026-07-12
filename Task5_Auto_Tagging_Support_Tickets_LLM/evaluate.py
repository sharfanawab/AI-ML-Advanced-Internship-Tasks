import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


# Load prediction files

zero_df = pd.read_csv(
    "results/zero_shot_predictions.csv"
)

few_df = pd.read_csv(
    "results/few_shot_predictions.csv"
)


# Take first predicted tag

def first_tag(value):
    return value.split(",")[0].strip()


zero_predictions = zero_df["predicted_tags"].apply(
    first_tag
)

few_predictions = few_df["few_shot_tags"].apply(
    first_tag
)


# Actual categories

actual = zero_df["category"]


# Accuracy

zero_accuracy = accuracy_score(
    actual,
    zero_predictions
)

few_accuracy = accuracy_score(
    actual,
    few_predictions
)


# F1 Score

zero_f1 = f1_score(
    actual,
    zero_predictions,
    average="weighted"
)

few_f1 = f1_score(
    actual,
    few_predictions,
    average="weighted"
)


print("Zero-shot Accuracy:", zero_accuracy)
print("Few-shot Accuracy:", few_accuracy)

print("Zero-shot F1 Score:", zero_f1)
print("Few-shot F1 Score:", few_f1)


# Save comparison

comparison = pd.DataFrame({

    "Method": [
        "Zero-shot",
        "Few-shot"
    ],

    "Accuracy": [
        zero_accuracy,
        few_accuracy
    ],

    "F1 Score": [
        zero_f1,
        few_f1
    ]

})


comparison.to_csv(
    "results/performance_comparison.csv",
    index=False
)


# Accuracy graph

plt.figure(figsize=(6,4))

plt.bar(
    comparison["Method"],
    comparison["Accuracy"]
)

plt.ylim(0,1)

plt.title(
    "Zero-shot vs Few-shot Accuracy"
)

plt.ylabel(
    "Accuracy"
)

plt.savefig(
    "results/performance_comparison.png"
)

plt.close()


print("Evaluation completed successfully!")