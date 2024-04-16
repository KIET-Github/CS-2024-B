from flask import Flask, jsonify, render_template

# Import necessary libraries for machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import nltk

# Initialize Flask app
app = Flask(__name__)


# Load and preprocess the data
def load_and_preprocess_data():
    df = pd.read_csv("Mental_Health_FAQ.csv")
    df = df.drop(["Question_ID"], axis="columns")
    df["Questions"] = df["Questions"].str.lower()

    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')

    def data_prep(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = " ".join(t for t in text.split() if t not in stopwords.words('english'))
        return text

    df["Questions"] = df["Questions"].apply(data_prep)
    return df


# Train the machine learning model
def train_model(df):
    tf = TfidfVectorizer()
    tf_train = tf.fit_transform(df["Questions"])

    le = LabelEncoder()
    df["Answers_Code"] = le.fit_transform(df["Answers"])

    mn = MultinomialNB()
    mn.fit(tf_train, df["Answers_Code"])

    return tf, le, mn


# Define the home route with a custom message
@app.route('/')
def home():
    return render_template('index.html', message='Welcome to the Mental Health FAQ API')


# Define the API route for answering questions
@app.route('/api/FAQ/<question>', methods=['GET'])
def answer_question(question):
    df = load_and_preprocess_data()
    tf, le, mn = train_model(df)

    testing = tf.transform([question])
    prediction = mn.predict(testing)[0]

    answers = df["Answers"].unique()

    # Use prediction to directly access the answer
    answer = answers[prediction]

    # Create a response dictionary
    response = {
        "answer": answer
    }

    # Convert the response dictionary to JSON and return
    return jsonify(response['answer'])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
