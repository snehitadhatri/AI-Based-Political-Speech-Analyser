import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the dataset (replace with your dataset path)
try:
    df = pd.read_csv("political_bias.csv")  # Example filename
except FileNotFoundError:
    print("Error: Dataset file not found. Please ensure the path is correct.")
    exit()

# Preprocessing (adjust based on your dataset's columns)
print(df.columns)
print(df["Text"].isnull().sum())
df["Text"] = df['Text'].fillna('')
if 'Text' not in df.columns or 'Bias' not in df.columns:
    print("Error: Required columns 'Text' or 'Bias' not found in the dataset.")
    exit()

df = df.dropna(subset=['Text', 'Bias'])  # Remove rows with missing values
X = df['Text']
y = df['Bias']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Training (Logistic Regression)
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train_vec, y_train)

# Model Evaluation
y_pred = model.predict(X_test_vec)

# Output Results with additional metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", model.score(X_test_vec, y_test))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Topic Modeling (LDA)
count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X_counts = count_vectorizer.fit_transform(X)

n_topics = 5  # Number of topics to extract
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X_counts)

def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-no_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics.append(top_features)
    return topics

topics = display_topics(lda, count_vectorizer.get_feature_names_out(), 10)
print("Extracted Topics:")
for idx, topic in enumerate(topics):
    print(f"Topic {idx + 1}: {', '.join(topic)}")

# Sentiment Polarity Scoring
def sentiment_score(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Example prediction function with topic and sentiment
def predict_bias_with_analysis(speech):
    speech_vec = vectorizer.transform([speech])
    prediction = model.predict(speech_vec)[0]
    sentiment = sentiment_score(speech)
    speech_count_vec = count_vectorizer.transform([speech])
    topic_distribution = lda.transform(speech_count_vec)[0]
    return {
        'bias': prediction,
        'sentiment': sentiment,
        'topic_distribution': topic_distribution.tolist()
    }

example_speech = "The other party is completely wrong and is destroying our country. We must take action against them."
print(f"Example Speech: {example_speech}")
print(f"Predicted Bias with Analysis: {predict_bias_with_analysis(example_speech)}")

example_speech2 = "We must work together to find common ground and solve our problems."
print(f"Example Speech: {example_speech2}")
print(f"Predicted Bias with Analysis: {predict_bias_with_analysis(example_speech2)}")
