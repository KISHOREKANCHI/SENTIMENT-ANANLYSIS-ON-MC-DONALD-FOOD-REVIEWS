#libraries
import numpy as np
import pandas as pd
# from google.colab import drive
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
import joblib


stopwords_set = set(stopwords.words('english'))
def processing(text):

    # Step 1: Remove Accented Characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Step 2: Tokenization
    tokens = word_tokenize(text)

    # Step 3: Stopwords Removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Step 4: Remove Numbers and Extra Whitespaces
    filtered_tokens = [re.sub(r'\d+', '', word) for word in filtered_tokens]
    filtered_tokens = [word.strip() for word in filtered_tokens if word.strip()]

    # Step 5: Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Step 6: Stemming (optional)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]

    # Step 7: Remove Single Letters
    filtered_tokens = [word for word in stemmed_tokens if len(word) > 1]

    # Join the tokens back into a clean text string
    clean_text = ' '.join(filtered_tokens)


    return clean_text



loaded_model = joblib.load('ensemble_model.joblib')
loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')
Input=input("Enter a review : ")
input_vector = loaded_vectorizer.transform([processing(Input)])
predicted_rating = loaded_model.predict(input_vector)
print("Predicted Rating:", predicted_rating)