"""
app.py -- Streamlit demo for the email/SMS spam classifier.

Run:
    streamlit run app.py
"""

import streamlit as st

from spam_classifier import SpamClassifier

st.set_page_config(page_title="Spam Classifier", page_icon="\U0001F4E7", layout="centered")


@st.cache_resource(show_spinner="Training classifier...")
def get_classifier() -> SpamClassifier:
    return SpamClassifier()


st.title("Email / SMS Spam Classifier")
st.caption(
    "NLP + KNN classifier trained on a labeled ham/spam dataset "
    "(CountVectorizer text vectorization, scikit-learn KNN)."
)

clf = get_classifier()
st.metric("Held-out test accuracy", f"{clf.accuracy * 100:.1f}%")

message = st.text_area(
    "Paste an email or SMS message",
    placeholder="Congratulations! You've won a free prize, click here to claim...",
    height=150,
)

if st.button("Classify", type="primary") and message.strip():
    label = clf.predict(message)
    if label == "spam":
        st.error("This looks like **SPAM**.")
    else:
        st.success("This looks like a legitimate message (**ham**).")
