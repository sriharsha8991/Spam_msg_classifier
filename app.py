import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Streamlit application layout
st.title('Spam SMS Classifier')
st.write('Enter a message to check if it is spam or not:')

# Text input
user_input = st.text_area("Message", height=150)

if st.button('Classify'):
    
    processed_input = vectorizer.transform([user_input])
    prediction = model.predict(processed_input)
    if prediction[0] == 1:
        st.write('This message is likely **Spam**.')
    else:
        st.write('This message is likely **Not Spam**.')

# Run this app with: streamlit run app.py
