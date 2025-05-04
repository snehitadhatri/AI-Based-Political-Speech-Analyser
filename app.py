from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import os

app = Flask(__name__, static_folder='')

# Load dataset and train model on startup (or load pre-trained models if available)
df = pd.read_csv("political_bias.csv")
df["Text"] = df["Text"].fillna("")
df = df.dropna(subset=["Text", "Bias"])
X = df["Text"]
y = df["Bias"]

# Vectorizers and model training
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X_counts = count_vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_counts)

def sentiment_score(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

@app.route("/")
def home():
    return send_from_directory('', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    # serve static files like speech.js
    return send_from_directory('', path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    speech_vec = vectorizer.transform([text])
    prediction = model.predict(speech_vec)[0]

    sentiment = sentiment_score(text)

    speech_count_vec = count_vectorizer.transform([text])
    topic_distribution = lda.transform(speech_count_vec)[0]

    response = {
        "bias": prediction,
        "sentiment": sentiment,
        "topic_distribution": topic_distribution.tolist()
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run()
