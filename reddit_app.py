# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from flask import Flask, request, render_template, redirect, jsonify
# import _pickle as pickle
from pickle import load
import sklearn
import requests

loaded_model = load(open('reddit_model_nc.pkl', 'rb'))
app = Flask(__name__)
@app.route("/test")
def test():
    # create the json object to send to the prediction function
    title = 'This is the title of my test post'
    text = "I've always wanted to post to Reddit, but I never know where!  Please tell me!"
    content = {'title': title, 'text': text}
    # send a post request to that prediction function
    # note: this is hardcoded for localhost and will probably not work when deployed
    url = 'http://localhost:5000/predict.json'
    r = requests.post(url, json=content)
    requests
    # display the response
    return r.text
@app.route("/predict.json", methods=["POST"])
def predict():
    #> {'title': 'example title', 'text': 'Example reddit post text here'}
    request_data = request.get_json(force=True)
    title = request_data['title']
    text = request_data['text']
    # concatenate title and text, passed in as one variable to the model
    post = title + ' ' + text
    # get predictions, store as a Pandas Series
    preds = pd.Series(loaded_model.predict_proba([post])[0])
    # assign the subreddit classes to the index
    preds.index = loaded_model.classes_
    # sort by values to get the top results
    preds = preds.sort_values(ascending=False)
    # return the top 5 results as JSON
    return jsonify(subreddits=preds.index[:5].to_list(),
                    probabilities=preds[:5].to_list())
