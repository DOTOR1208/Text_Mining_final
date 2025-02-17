import pandas as pd
import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

# Load sentiment Hugging Face
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

data = pd.read_csv("crawl_movies.csv")

def classify_sentiment(text):
    text = str(text)[:512]  
    result = sentiment_model(text)[0]
    
    return "Positive" if result["label"] == "POSITIVE" else "Negative"

# Sentiment of "content"
data["sentiment"] = data["content"].apply(classify_sentiment)

data.to_csv("labeled_reviews.csv", index=False)
