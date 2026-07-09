"""
analysis.py -- Exploratory data analysis and visualization for the spam
classification dataset. Trains the same pipeline as spam_classifier.py and
renders a class-balance pie chart, spam/ham word clouds, and a confusion
matrix heatmap.

Run directly:
    python analysis.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from spam_classifier import SpamClassifier


def plot_class_balance(df):
    values = df["target"].value_counts()
    plt.figure()
    plt.pie(values, labels=["ham", "spam"], autopct="%1.2f%%", startangle=90)
    plt.title("Email Classification Balance")
    plt.savefig("class_balance.png", bbox_inches="tight")
    plt.close()


def plot_wordclouds(df):
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")

    spam_text = df[df["target"] == 1]["transformed_email"].str.cat(sep=" ")
    if spam_text:
        spam_wc = wc.generate(spam_text)
        plt.figure(figsize=(15, 6))
        plt.imshow(spam_wc)
        plt.axis("off")
        plt.title("Spam Word Cloud")
        plt.savefig("spam_wordcloud.png", bbox_inches="tight")
        plt.close()

    ham_text = df[df["target"] == 0]["transformed_email"].str.cat(sep=" ")
    if ham_text:
        ham_wc = wc.generate(ham_text)
        plt.figure(figsize=(15, 6))
        plt.imshow(ham_wc)
        plt.axis("off")
        plt.title("Ham Word Cloud")
        plt.savefig("ham_wordcloud.png", bbox_inches="tight")
        plt.close()


def plot_confusion_matrix(cm):
    plt.figure()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        annot_kws={"size": 16}, linewidths=0.1, linecolor="gray",
    )
    plt.title("Confusion Matrix", fontsize=18)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("Actual Labels", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()


def main():
    clf = SpamClassifier()
    print(f"Accuracy: {clf.accuracy:.4f}")

    plot_class_balance(clf.dataframe)
    plot_wordclouds(clf.dataframe)
    plot_confusion_matrix(clf.confusion_matrix)
    print("Saved class_balance.png, spam_wordcloud.png, ham_wordcloud.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
