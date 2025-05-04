---

# Truth Seeker AI – Political Speech Analyzer

Truth Seeker AI is a web-based tool that uses Machine Learning and Natural Language Processing (NLP) to analyze political speeches and detect potential bias. Built as part of a Data Science internship at Prasunet Pvt. Ltd., this tool bridges the gap between AI technology and civic understanding.

---

## Features

- Analyze speeches to detect *political bias* (Left, Right, or Neutral).
- Input speech text via HTML form.
- Real-time prediction using an integrated ML model.
- Intuitive and responsive front-end design.
- Model trained on political speech/text datasets using NLP and Logistic Regression.

---

## Tech Stack

- *Frontend:* HTML, CSS, JavaScript
- *Backend:* Python (Flask)
- *ML/NLP:* scikit-learn, pandas, NumPy, TF-IDF, Logistic Regression
- *Deployment:* Localhost / Web Server

---

## How It Works

1. User inputs a political speech into the web interface.
2. The text is preprocessed and vectorized using a TF-IDF vectorizer.
3. A Logistic Regression model analyzes the text for political bias.
4. The result is returned to the user on the same page with a probability score.

---

## Project Structure

├── static/ │   └── style.css ├── templates/ │   └── index.html            # Truth Seeker UI ├── model/ │   ├── bias_model.pkl        # Trained logistic regression model │   └── tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer ├── app.py                    # Flask backend logic ├── preprocess.py             # Text cleaning & NLP functions ├── requirements.txt └── README.md

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/truth-seeker-ai.git
cd truth-seeker-ai
```
### 2. Create Virtual Environment
```
python -m venv venv
venv\Scripts\activate      # For Windows
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 4. Run the Application
```
python app.py
```
Then open your browser and navigate to:
http://127.0.0.1:5000


---

### Future Improvements

Add multilingual speech analysis support.

Expand model with larger, more diverse political datasets.

Integrate sentiment analysis alongside bias detection.

Deploy on cloud platforms (Render, Heroku, AWS).



---

### Author

Snehita Dhatri Siddabattuni
Data Science Intern – Prasunet Pvt. Ltd.


---

### License

This project is for educational and research purposes.
