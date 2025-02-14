# Email Spam Classification Using Machine Learning  

This project implements an **Email Spam Classifier** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. The dataset is processed, transformed, and classified into **ham (legitimate emails)** and **spam (unwanted emails)** using a **K-Nearest Neighbors (KNN) classifier**.  

## Features  
- **Data Preprocessing**:  
  - Handling missing and duplicate values  
  - Converting text to lowercase  
  - Tokenization and special character removal  
  - Stopword filtering  
- **Data Visualization**:  
  - WordCloud for spam and ham emails  
  - Pie chart for data distribution  
  - Confusion matrix for model evaluation  
- **Machine Learning Implementation**:  
  - **Text Vectorization** using CountVectorizer  
  - **Model Training & Testing** using KNN  
  - **Performance Evaluation** (Accuracy & Confusion Matrix)  

## Dependencies  
- Python  
- Pandas, NumPy  
- NLTK (Natural Language Toolkit)  
- Matplotlib, Seaborn  
- Scikit-learn  

## How to Run  
1. Install dependencies:  
   ```bash
   pip install numpy pandas nltk wordcloud matplotlib seaborn scikit-learn
   ```  
2. Download and load the dataset (`email_dataset.csv`).  
3. Run the script to preprocess data and train the model.  
4. Evaluate performance using accuracy score and confusion matrix.  

## Results  
The trained model effectively classifies spam and ham emails, providing insights into email filtering using **KNN-based classification**. Further improvements can be made using **TF-IDF, deep learning models, or ensemble learning techniques**.  

#machinelearning #nlp #spamclassification #datascience #knn
