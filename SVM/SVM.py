import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2


# Initialize linguistic tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Enhanced preprocessing with negation handling
def preprocess_text(text):
    # Text normalization
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)        # HTML tags
    text = re.sub(r'https?://\S+', ' ', text)  # URLs
    text = re.sub(r"\b\d+\b", ' ', text)       # Standalone numbers
    text = re.sub(r"[^\w\s']", ' ', text)      # Keep apostrophes
    
    # Handle contractions
    contractions = {
        "n't": " not", "'s": " is", "'re": " are",
        "'ve": " have", "'ll": " will", "'d": " would"
    }
    for pat, repl in contractions.items():
        text = text.replace(pat, repl)
    
    # Tokenization and lemmatization
    tokens = nltk.word_tokenize(text)
    processed = []
    negation = False
    
    for word in tokens:
        word = lemmatizer.lemmatize(word)
        if word in ["not", "no"]:
            negation = True
            continue
        elif negation:
            processed.append(f"not_{word}")
            negation = False
        else:
            if word not in stop_words and len(word) > 2:
                processed.append(word)
    
    return ' '.join(processed)

# Load data
train_df = pd.read_csv('C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\train_clean.csv')
test_df = pd.read_csv('C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\test_clean.csv')

# Apply preprocessing
print("Preprocessing data...")
train_df['clean_text'] = train_df['reviews'].apply(preprocess_text)
test_df['clean_text'] = test_df['reviews'].apply(preprocess_text)

# Prepare data
X_train = train_df['clean_text']
y_train = train_df['rating']
X_test = test_df['clean_text']
y_test = test_df['rating']

# Optimized TF-IDF with feature selection
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=200000,
    min_df=3,
    max_df=0.7,
    sublinear_tf=True,
    norm='l2'
)

print("Vectorizing text...")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Feature selection
print("Selecting best features...")
selector = SelectKBest(chi2, k=125000)
X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = selector.transform(X_test_tfidf)

# Precision tuning
param_dist = {
    'C': loguniform(1e-2, 1e2),
    'loss': ['squared_hinge', 'hinge'],
    'class_weight': ['balanced', None]
}

print("Training optimized model...")
model = RandomizedSearchCV(
    LinearSVC(
        class_weight='balanced',
        max_iter=40000,
        dual=True  # Explicitly set to suppress warning
    ),
    param_dist,
    n_iter=40,
    cv=2,
    scoring='accuracy',
    n_jobs=2,
    random_state=42
)
model.fit(X_train_selected, y_train)

# Evaluate
print("\n=== Final Results ===")
y_pred = model.predict(X_test_selected)
print(f"Best Parameters: {model.best_params_}")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save predictions
pd.DataFrame({
    'text': test_df['reviews'],
    'predicted': y_pred
}).to_csv('improved_predictions.csv', index=False)