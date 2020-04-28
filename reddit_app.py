# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from flask import Flask, request, render_template, redirect, jsonify
# import _pickle as pickle
from pickle import load
loaded_model = load(open('reddit_model.pkl', 'rb'))
application = app = Flask(__name__)
@app.route("/")
def home():
    title = 'this is a title'
    text = 'this is some text'
    return redirect('/predict.json')
@app.route("/predict.json", methods=["GET", "POST"])
def predict():
  #> {'title': 'example title', 'text': 'Example reddit post text here'}
  # concatenate title and text, passed in as one variable to the model
  post = "this is a post"
#   post = request.form["title"] + ' ' + request.form["text"]
  # get predictions, store as a Pandas Series
  #preds = pd.Series(loaded_model.predict_proba(post)[0])
  # assign the subreddit classes to the index
  #preds.index = loaded_model.classes_
  # sort by values to get the top results
  #preds = preds.sort_values(ascending=False)
  # return the top 5 results as JSON
  preds = pd.Series([.5, .4, .3, .2, .1])
  preds.index = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5']
  return jsonify(subreddits=preds.index[:5].to_list(),
                  probabilities=preds[:5].to_list())
