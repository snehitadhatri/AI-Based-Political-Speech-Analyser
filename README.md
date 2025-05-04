# Truth Seeker

Truth Seeker is a Flask-based web application that detects political bias in text. It uses machine learning models trained on a political bias dataset to predict the bias of input text, along with sentiment analysis and topic distribution.

## Features

- Predict political bias of input text
- Sentiment analysis of the text
- Topic distribution using Latent Dirichlet Allocation (LDA)
- Simple web interface and API endpoints

## Setup and Installation

1. Clone the repository and navigate to the `truth_seeker` directory.

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure the dataset file `Political_Bias.csv` is present in the `truth_seeker` directory.

## Usage

### Running Locally

To run the app locally for development or testing, use:

```bash
python app.py
```

This will start the Flask development server. You can access the app at `http://127.0.0.1:5000/`.

### API Endpoints

- `GET /` - Serves the main index.html page.
- `POST /predict` - Accepts JSON with a "text" field and returns the predicted bias, sentiment score, and topic distribution.

Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"Your input text here\"}"
```

## Deployment

The project includes a `Procfile` for deployment on platforms like Heroku.

### Deploying on Heroku

1. Ensure you have the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed and are logged in.

2. Create a new Heroku app:

```bash
heroku create your-app-name
```

3. Push the code to Heroku:

```bash
git push heroku main
```

4. Heroku will use the `Procfile` to run the app with Gunicorn:

```
web: gunicorn app:app
```

5. Open the deployed app:

```bash
heroku open
```

### Notes

- The `Procfile` specifies the command to run the Flask app using Gunicorn.
- Make sure all dependencies are listed in `requirements.txt`.
- The app expects the `Political_Bias.csv` dataset to be present in the root directory.

## License

This project is licensed under the MIT License.
