# from flask import Flask
#
# app = Flask(__name__)
#
#
# @app.route('/')
# def hello_world():
#     return 'Hello World!'
#
#
# @app.route('/api/FAQ/<question>', methods=['GET'])
# def index(question):  # put application's code here
#     from flask import jsonify
#     from sklearn.preprocessing import LabelEncoder
#     from sklearn.naive_bayes import MultinomialNB
#     from nltk.corpus import stopwords
#     from nltk.stem import WordNetLemmatizer
#     from nltk.tokenize import word_tokenize
#     from nltk.tokenize import RegexpTokenizer
#     from sklearn.feature_extraction.text import CountVectorizer
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     import pandas as pd
#     import numpy as np
#     import string
#     import re
#     import nltk
#     df = pd.read_csv("Mental_Health_FAQ.csv")
#     # df.info()
#     # df.head()
#     nltk.download('wordnet')
#     nltk.download('stopwords')
#     nltk.download('punkt')
#     df = df.drop(["Question_ID"], axis="columns")
#     df["Questions"] = df["Questions"].str.lower()
#     text = df["Questions"][0]
#     text = re.sub(r'[^\w\s]', '', text)
#     words = word_tokenize(text)
#     # print(words)
#     wnl = WordNetLemmatizer()
#     text = " ".join([wnl.lemmatize(t) for t in text.split()])
#     stopwords = stopwords.words('english')
#
#     def data_prep(text):
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', '', text)
#         text = " ".join(t for t in text.split() if t not in stopwords)
#         return text
#
#     # print(data_prep(text))
#     df["Questions"] = df["Questions"].apply(data_prep)
#     # print(df["Questions"][0])
#     tf = TfidfVectorizer()
#     # transformed train reviews
#     tf_train = tf.fit_transform(df["Questions"])
#     # #transformed test reviews
#     # tf_test=tf.transform(test_reviews_data)
#     # print('Tfidf_train:',tf_train.shape)
#     # print('Tfidf_test:',tf_test.shape)
#     df_check = pd.DataFrame(tf_train.toarray(), columns=tf.get_feature_names_out())
#
#     le = LabelEncoder()
#     df["Answers_Code"] = le.fit_transform(df["Answers"])
#
#     mn = MultinomialNB()
#     mn.fit(df_check, df["Answers_Code"])
#     test = [question]
#     testing = tf.transform(test)
#     # print(mn.predict(testing))
#     Ans = df["Answers"].unique()
#     Ans = list(Ans)
#     Ans_Code = df["Answers_Code"].unique()
#     Ans_Code = Ans_Code.tolist()
#     Ans_Code.index(48)
#     Ans_Code.index(36)
#     return jsonify(Ans[mn.predict(testing)[0]])
#
#
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)
