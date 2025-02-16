import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download package (run once then command)
# nltk.download('vader_lexicon')

data = pd.read_csv("crawl_movies.csv")  # Ensure the file contains columns: movie_id, author, content, url, source

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(str(text))["compound"]  # Convert to string to avoid NaN errors
    if score >= 0.7:
        return "Super Positive"
    elif score >= 0.05:
        return "Positive"
    elif score > -0.7:
        return "Negative"
    else:
        return "Super Negative"

# Apply sentiment analysis
data["sentiment"] = data["content"].apply(get_sentiment)

data.to_csv("labeled_reviews.csv", index=False)

print("Sentiment labeling completed with 4 levels using VADER!")
