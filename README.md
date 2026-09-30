# Fake News Classifier 

##  Project Overview

This project is a classifier model built to classify news articles as 'Real' or 'Fake' using **Natural Language Processing (NLP)** techniques. The model utilizes machine learning and deep learning approaches to distinguish between real and fake news articles accurately.

##  Models & Performance

The project tests various models and feature extraction techniques. The top-performing models are:

| Model              | Feature Extraction | Accuracy |
|-------------------|-------------------|----------|
| Support Vector Machine (SVM) | TF-IDF | **99.45%** |
| Kim's CNN | Word2Vec | **99.89%** |

## Dataset Details

The dataset is stored in the `dataset/` folder and includes the following columns:

* label: 0 for Fake news, 1 for Real news.

* title: The headline of the article.

* text: The full content of the article.

* subject: The category of the article (e.g., Politics, Technology, etc.).

* date: The publication date of the article.

## Project Files

Here are the main files included in this project:

-   `Streamlit_app.py` - Web app script using **Streamlit** to classify news.
    
-   `PROJECT_NLP_Challenge.ipynb` - Jupyter Notebook with model training and evaluation.
    
-   `cnn_model.keras` - Trained **Kim's CNN model** for classification.
    
-   `fake_news_classifier_svc.pkl` - Trained **SVM model** for detecting fake news.
    
-   `tokenizer.pkl` - Pretrained **Word2Vec tokenizer** for text preprocessing.
    
-   `tfidf_vectorizer.pkl` - Fitted **TF-IDF Vectorizer** for text features.
    
-   `requirements.txt` - List of dependencies required.
    
-   `dataset/` - Folder containing labeled news articles.

## Live Demo

You can try the Fake News Classifier online here: 👉 [Live Demo](https://nlp-neurallen.streamlit.app/)
