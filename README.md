# Email/SMS Spam Classifier

NLP + machine learning spam classifier: preprocesses a labeled ham/spam dataset, vectorizes it with `CountVectorizer`, and classifies messages with a K-Nearest Neighbors model. Includes an interactive Streamlit demo.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-KNN-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

**[Live demo →](#deployment)**

## Features

- **Text preprocessing** — lowercasing, tokenization, and alphanumeric filtering (NLTK).
- **Vectorization** — `CountVectorizer` bag-of-words representation.
- **Classification** — K-Nearest Neighbors (k=3), evaluated on a held-out 20% test split.
- **Visualization** (`analysis.py`) — class-balance pie chart, spam/ham word clouds, and a confusion-matrix heatmap.
- **Interactive demo** (`app.py`) — paste a message and get an instant spam/ham prediction with the model's held-out accuracy shown live.

## Tech Stack

Python, pandas, NLTK, scikit-learn, WordCloud, Matplotlib/Seaborn, Streamlit.

## How to Run

```bash
# Clone
git clone https://github.com/Abdelrahmanemad0/Spam-or-not-spam.git
cd Spam-or-not-spam

# Install dependencies
pip install -r requirements.txt

# Launch the interactive demo
streamlit run app.py

# Or run the EDA/visualization script (saves PNGs to the working directory)
python analysis.py
```

## Project Structure

- `spam_classifier.py` — core pipeline: dataset loading, preprocessing, training, and a `SpamClassifier.predict()` API shared by `app.py` and `analysis.py`
- `app.py` — Streamlit demo UI
- `analysis.py` — EDA and visualization (class balance, word clouds, confusion matrix)
- `email_dataset.csv` — labeled ham/spam dataset

## Deployment

Deployable in minutes on **Streamlit Community Cloud**: fork this repo, connect it at [share.streamlit.io](https://share.streamlit.io), and set `app.py` as the entry point. No secrets required — the model trains from the bundled CSV at startup.

## Results & Future Improvements

The KNN baseline classifies spam/ham using simple bag-of-words features. Documented next steps: swap in TF-IDF weighting, try an ensemble (e.g. Naive Bayes or gradient boosting), and evaluate on a larger, more diverse dataset.

## License

MIT — see [LICENSE](LICENSE).

#machinelearning #nlp #spamclassification #datascience #knn
