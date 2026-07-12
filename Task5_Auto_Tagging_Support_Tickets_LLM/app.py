import streamlit as st
from transformers import pipeline


# Title

st.title(" Support Ticket Auto Tagging Using LLM")


st.write(
    "Enter a support ticket and get top 3 predicted tags"
)


# Load model

@st.cache_resource
def load_model():

    model = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    return model


classifier = load_model()


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


# User input

ticket = st.text_area(
    "Enter Support Ticket"
)


# Button

if st.button("Predict Tags"):

    if ticket:

        result = classifier(
            ticket,
            labels,
            multi_label=True
        )


        st.subheader(
            "Top 3 Predicted Tags"
        )


        for label, score in zip(
            result["labels"][:3],
            result["scores"][:3]
        ):

            st.write(
                f"**{label}** : {score:.3f}"
            )

    else:

        st.warning(
            "Please enter a ticket"
        )