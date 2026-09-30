import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load trained CNN model
model = load_model("cnn_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Constants
MAX_SEQUENCE_LENGTH = 100

# Streamlit UI
st.title("Fake News Classifier using CNN")
st.write("Enter a news article title and text to check if it's Fake or Real.")

# Input fields
title = st.text_input("Title", "")
text = st.text_area("News Text", "")

if st.button("Classify"):
    if title and text:
        # Combine title and text
        input_text = title + " " + text
        
        # Preprocess input text
        sequence = tokenizer.texts_to_sequences([input_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded_sequence)
        predicted_class = np.argmax(prediction)
        
        # Map prediction to labels
        label = "Fake News" if predicted_class == 0 else "Real News"
        
        st.subheader(f"Prediction: {label}")
    else:
        st.warning("Please enter both a title and text.")
