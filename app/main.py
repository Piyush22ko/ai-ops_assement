
import os
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.joblib"))

app = FastAPI(title="Email Spam Classifier")


class EmailInput(BaseModel):
    email_text: str = Field(min_length=1)


@app.get("/")
def home():
    return {"message": "Email Spam Classifier API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: EmailInput):
    email_vector = vectorizer.transform([data.email_text])
    prediction = str(model.predict(email_vector)[0])

    label = prediction.strip().lower()

    if label in ("spam", "1"):
        result = "Spam"
    elif label in ("ham", "not spam", "0"):
        result = "Not Spam"
    else:
        result = prediction

    return {
        "prediction": prediction,
        "result": result
    }