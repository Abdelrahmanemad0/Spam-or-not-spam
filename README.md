# Email Spam Classification Using Machine Learning

This project implements an email/SMS spam classifier using Natural Language Processing (NLP) and Machine Learning. The dataset is cleaned, transformed, and classified into **ham** (legitimate messages) and **spam** (unwanted messages) using a K-Nearest Neighbors (KNN) classifier.

## Features

- **Data Preprocessing** — handling missing/duplicate values, lowercasing, tokenization, special-character removal, stopword filtering
- **Data Visualization** — WordCloud for spam vs. ham messages, pie chart of the class distribution, confusion matrix for model evaluation
- **Machine Learning** — text vectorization with `CountVectorizer`, training/testing with KNN, accuracy and confusion-matrix evaluation

## Dependencies

- Python
- Pandas, NumPy
- NLTK (Natural Language Toolkit)
- WordCloud
- Matplotlib, Seaborn
- Scikit-learn

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas nltk wordcloud matplotlib seaborn scikit-learn
   ```
2. Make sure `email_dataset.csv` is in the same directory as the script (it's included in this repo).
3. Run the script:
   ```bash
   python spam_classifier.py
   ```
   This loads the dataset, cleans it, trains the KNN classifier, prints the accuracy, and displays the pie chart, word clouds, and confusion matrix as separate windows.

## Project Structure

- `spam_classifier.py` — main script (renamed from `Spam or Not Spam`, which had no file extension)
- `email_dataset.csv` — labeled dataset of spam/ham messages
- `README.md` — this file
- `LICENSE` — MIT license
- `.gitignore` — Python build/editor artifacts to keep out of version control

## Fixes in this revision

- **Renamed `Spam or Not Spam` to `spam_classifier.py`** — the file had no extension and couldn't be run directly with `python`.
- **The script referenced `df` before it was ever loaded** — it was a raw Jupyter/Colab cell dump missing the `pd.read_csv(...)` call. Added `df = pd.read_csv('email_dataset.csv', encoding='latin-1')` at the top.
- **Removed the invalid `%matplotlib inline` line**, which is IPython-only syntax and would raise a `SyntaxError` in a plain `.py` file (left as a commented-out note for notebook users).
- **Actually implemented the stopword filtering the README always claimed as a feature.** `transform_text()` tokenized and removed non-alphanumeric characters, but never filtered stopwords despite the comment above it saying it did. It now filters against NLTK's English stopword list.

## Results

The trained model classifies spam and ham messages, printing accuracy and a confusion matrix. Further improvements could use TF-IDF, deep learning models, or ensemble methods.

## License

MIT — see [LICENSE](LICENSE).

#machinelearning #nlp #spamclassification #datascience #knn
