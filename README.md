# VITYARTHI-FUNDAMENTS-IN-AI-ML-PROJECT
📘 Fake News Detection Using Machine Learning

A Machine Learning project that detects whether a news article is Real or Fake using Natural Language Processing (NLP) techniques.
This project uses TF-IDF Vectorization and Logistic Regression to classify news articles with high accuracy.

🚀 Project Overview

Fake news spreads rapidly through social media and online platforms. This project aims to automatically identify fake news by analyzing the text content of articles.

The model:

Cleans and preprocesses text

Converts it into numerical form

Trains a classifier

Predicts whether the input article is REAL or FAKE

🧠 Features

Text cleaning (lowercasing, punctuation removal, stopword removal, lemmatization)

TF-IDF text vectorization

Logistic Regression classifier

High accuracy (≈93% depending on dataset)

Command-line prediction tool

Model saving using joblib

📂 Project Structure
Fake-News-Detection/
│
├── fake_news_detection.py     # Main ML script
├── news.csv                   # Dataset (user-provided)
├── fake_news_model.joblib     # Saved ML model (generated after training)
├── tfidf_vectorizer.joblib    # Saved vectorizer (generated)
├── README.md                  # Project documentation
└── Fake_News_Detection_Report.pdf   # Project report

📊 Dataset

You can download a dataset from:

Kaggle: Fake News Detection Dataset

Or any dataset with:

text → news content

label → 0 = Real, 1 = Fake

Name your dataset file: news.csv
Or change the path inside fake_news_detection.py.

🛠️ Technologies Used

Python

NumPy

Pandas

Scikit-learn

NLTK

Joblib

TF-IDF Vectorizer

Logistic Regression

🔧 How to Run the Project
1️⃣ Install Required Libraries
pip install pandas numpy scikit-learn nltk joblib

2️⃣ Place Your Dataset

Add news.csv in the project folder.

3️⃣ Run the Script
python fake_news_detection.py

4️⃣ Test Your Own News

After training, the script will open a small input console:

> Enter news text:
"The government announced a new plan..."


Output example:

Predicted label: REAL
P(fake)=0.12, P(real)=0.88

📈 Model Details
Algorithm Used

Logistic Regression

Works great for binary text classification

Fast and interpretable

Text Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency)

Removes stopwords

Gives weight to important words

🧪 Results

The model achieves:

Accuracy: ~93%

Precision/Recall: High

Confusion Matrix: Low misclassification

Performance may vary with dataset quality.

📌 Applications

Fake news filters on social media

Browser plugins

News verification tools

Journalism research

Content moderation systems

🚧 Limitations

Works only for English text

Cannot detect sarcasm or satire

Cannot analyze images or videos

Dataset heavily affects performance

🔮 Future Enhancements

Use Deep Learning (LSTM, BERT)

Build a web app (Flask / Streamlit)

Multi-language support

Browser extension for real-time detection

👤 Author

AYUSH SHUKLA
B.Tech (1st Year) – Artificial Intelligence / Computer Science
Email = ayush.25bai10146@vitbhopal.ac.in
