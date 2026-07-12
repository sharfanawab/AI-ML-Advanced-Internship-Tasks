import pandas as pd
from transformers import pipeline


# Load dataset

df = pd.read_csv(
    "data/support_tickets.csv"
)


# Load LLM

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


print("Few-shot model loaded successfully")


# Few-shot examples

examples = """
Example 1:
Ticket: My internet connection is not working
Tag: Internet Issue

Example 2:
Ticket: I forgot my password and cannot login
Tag: Account Login

Example 3:
Ticket: My payment was declined
Tag: Payment Issue

Example 4:
Ticket: Mobile application crashes
Tag: App Problem
"""


# Labels

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


# Few-shot function

def few_shot_predict(ticket):

    prompt = examples + "\nTicket: " + ticket


    result = classifier(
        prompt,
        labels,
        multi_label=True
    )


    return ", ".join(
        result["labels"][:3]
    )


# Apply

df["few_shot_tags"] = df["ticket_text"].apply(
    few_shot_predict
)


# Save

df.to_csv(
    "results/few_shot_predictions.csv",
    index=False
)


print("Few-shot predictions saved!")
print(df.head())