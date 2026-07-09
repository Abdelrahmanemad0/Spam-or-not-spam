"""
spam_classifier.py -- Core email/SMS spam classification pipeline.

Loads email_dataset.csv, preprocesses text, trains a CountVectorizer + KNN
classifier, and exposes a `SpamClassifier` class other scripts (analysis.py,
app.py) can reuse instead of duplicating the pipeline.
"""

from __future__ import annotations

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATASET_PATH = "email_dataset.csv"


def ensure_nltk_resources() -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load and clean the raw email dataset."""
    df = pd.read_csv(path)
    df = df.rename(columns={"v1": "target", "v2": "email"})
    df = df.drop_duplicates()
    df = df.dropna(subset=["target", "email"])
    return df


def transform_text(text: str) -> str:
    """Lowercase, tokenize, and strip non-alphanumeric tokens."""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    return " ".join(tokens)


class SpamClassifier:
    """Trains a CountVectorizer + KNN spam classifier on init and exposes
    `predict`/`predict_proba`-style helpers for a single message."""

    def __init__(self, dataset_path: str = DATASET_PATH, n_neighbors: int = 3):
        ensure_nltk_resources()

        df = load_dataset(dataset_path)
        df["target"] = (df["target"].str.strip().str.lower() == "spam").astype(int)
        df["transformed_email"] = df["email"].apply(transform_text)

        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(df["transformed_email"]).toarray()
        y = df["target"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=2
        )

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        self.dataframe = df

    def predict(self, text: str) -> str:
        """Return 'spam' or 'ham' for a single email/SMS body."""
        processed = transform_text(text)
        vectorized = self.vectorizer.transform([processed]).toarray()
        prediction = self.model.predict(vectorized)[0]
        return "spam" if prediction == 1 else "ham"
