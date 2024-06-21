import os
os.chdir('D:\mini project')

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def load_model():
    with open('LogisticRegression.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_text(review, vectorizer):
    # Load the dataset or define df_all here
    df_all = pd.read_csv("D:\\mini project\\tripadvisor_hotel_reviews.csv")  # Update with your dataset path
    import re
    custom_stopwords = {'don',"don't",'ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'no','nor','not','shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"}
    corpus = []
    ps=PorterStemmer()
    stop_words=set(stopwords.words("english")) - custom_stopwords
    for i in range(len(df_all)):
        r1 = re.sub('[^a-zA-Z]', ' ', df_all['Review'][i])
        r1 = r1.lower()
        r1 = r1.split()
        r1 = [ps.stem(word) for word in r1 if word not in stop_words]
        r1 = " ".join(r1)
        corpus.append(r1)

    # Transform the review into numerical features
    review_transformed = vectorizer.transform([review])
    return review_transformed

def main():
    st.title("Hotel Review Sentiment Analysis")
    st.write("Enter your hotel review below:")

    review = st.text_area("Review")

    if st.button("Predict Sentiment"):
        if review:
            # Load the pre-trained model
            model = load_model()

            # Load or define the vectorizer
            try:
                with open('vectorizer.pkl', 'rb') as f:
                    vectorizer = pickle.load(f)
            except FileNotFoundError:
                vectorizer = CountVectorizer(max_features=2000)
                vectorizer.fit(df_all)
                with open('vectorizer.pkl', 'wb') as f:
                    pickle.dump(vectorizer, f)

            # Preprocess the text data
            review_transformed = preprocess_text(review, vectorizer)

            # Predict sentiment
            prediction = model.predict(review_transformed)

            # Display result
            if prediction == '1':
                st.write("Sentiment: Positive")
            else:
                st.write("Sentiment: Negative")
        else:
            st.write("Please enter a review before predicting.")

if __name__ == "__main__":
    main()
