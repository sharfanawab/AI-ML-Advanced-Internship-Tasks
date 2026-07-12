import pandas as pd
from transformers import pipeline


# Load dataset
df = pd.read_csv("data/support_tickets.csv")


# Load model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

print("Zero-shot model loaded successfully")


# Possible tags
labels = [
    "Internet Issue",
    "Account Login",
    "Payment Issue",
    "App Problem",
    "Subscription",
    "Technical Issue",
    "Billing Problem",
    "Feature Request"
]


# Function for top 3 tags

def get_top_tags(ticket):

    result = classifier(
        ticket,
        labels,
        multi_label=True
    )

    top_tags = result["labels"][:3]

    return ", ".join(top_tags)



# Apply on all tickets

df["predicted_tags"] = df["ticket_text"].apply(get_top_tags)


# Save result

df.to_csv(
    "results/zero_shot_predictions.csv",
    index=False
)


print("Top 3 tags generated successfully!")
print(df.head())