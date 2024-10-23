import nltk
nltk.download('stopwords') 

import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords # the for of in with
from nltk.stem.porter import PorterStemmer # loved loving == love
from sklearn.feature_extraction.text import TfidfVectorizer # loved = [0.0]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

news_df = pd.read_csv('train.csv/train.csv')


news_df = news_df.fillna(' ')

news_df['content'] = news_df['author']+" "+news_df['title']
X=news_df.drop('label',axis=1)
y=news_df['label']

# stemming
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_df['content'] = news_df['content'].apply(stemming)


X = news_df['content'].values
y = news_df['label'].values



vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=2)




model = LogisticRegression()
model.fit(X_train,y_train)

st.title("Fake News Detector")
input_text=st.text_input("Enter news Article")

# Predict whether input news is fake or real
if st.button("Check"):
    if input_text:
        # Preprocess the input text (similar to how training data was processed)
        processed_input = stemming(input_text)
        vectorized_input = vector.transform([processed_input])
        
        # Make prediction
        prediction = model.predict(vectorized_input)

        # Display result
        if prediction == 0:
            st.write("This news is Real.")
        else:
            st.write("This news is Fake.")

