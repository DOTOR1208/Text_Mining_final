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
import joblib

# Khởi tạo công cụ NLP
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)  # Xóa HTML tags
    text = re.sub(r'https?://\S+', ' ', text)  # Xóa URLs
    text = re.sub(r"\b\d+\b", ' ', text)  # Xóa số
    text = re.sub(r"[^\w\s']", ' ', text)  # Giữ dấu nháy đơn

    # Xử lý từ viết tắt
    contractions = {
        "n't": " not", "'s": " is", "'re": " are",
        "'ve": " have", "'ll": " will", "'d": " would"
    }
    for pat, repl in contractions.items():
        text = text.replace(pat, repl)

    # Tokenization và lemmatization
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

# Load dữ liệu
print("Loading data...")
train_df = pd.read_csv('C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\train_clean.csv')
test_df = pd.read_csv('C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\test_clean.csv')

# Tiền xử lý dữ liệu
print("Preprocessing data...")
train_df['clean_text'] = train_df['reviews'].apply(preprocess_text)
test_df['clean_text'] = test_df['reviews'].apply(preprocess_text)

# Chuẩn bị dữ liệu train và test
X_train = train_df['clean_text']
y_train = train_df['rating']
X_test = test_df['clean_text']
y_test = test_df['rating']

# Vector hóa văn bản với TF-IDF
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

# Chọn đặc trưng tốt nhất với chi-square test
selector = SelectKBest(chi2, k=125000)
X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = selector.transform(X_test_tfidf)

# Thiết lập tham số cho mô hình SVM
param_dist = {
    'C': loguniform(1e-2, 1e2),
    'loss': ['squared_hinge', 'hinge'],
    'class_weight': ['balanced', None]
}

# Train mô hình với Randomized Search
print("Training optimized model...")
model = RandomizedSearchCV(
    LinearSVC(
        class_weight='balanced',
        max_iter=40000,
        dual=True
    ),
    param_dist,
    n_iter=40,
    cv=2,
    scoring='accuracy',
    n_jobs=2,
    random_state=42
)
model.fit(X_train_selected, y_train)

# Đánh giá mô hình
print("\n=== Model Evaluation ===")
y_pred = model.predict(X_test_selected)
print(f"Best Parameters: {model.best_params_}")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Lưu mô hình, TF-IDF vectorizer và selector
print("Saving trained model...")
joblib.dump(model, 'C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\SVM\\svm_model.pkl')
joblib.dump(tfidf, 'C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\SVM\\tfidf_vectorizer.pkl')
joblib.dump(selector, 'C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\SVM\\feature_selector.pkl')
print("Model saved successfully!")

# ================================================
#       LOAD MODEL VÀ DỰ ĐOÁN TRÊN TẬP TEST
# ================================================

# Tải lại mô hình đã lưu
print("Loading saved model...")
model = joblib.load('C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\SVM\\svm_model.pkl')
tfidf = joblib.load('C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\SVM\\tfidf_vectorizer.pkl')
selector = joblib.load('C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\SVM\\feature_selector.pkl')
print("Model loaded successfully!")

# Dự đoán trên tập test
print("Predicting on test set...")
X_test_tfidf = tfidf.transform(X_test)
X_test_selected = selector.transform(X_test_tfidf)
y_pred = model.predict(X_test_selected)

# Lưu kết quả dự đoán
output_df = pd.DataFrame({
    'text': test_df['reviews'],
    'actual_rating': y_test,
    'predicted_rating': y_pred
})
output_df.to_csv('C:\\Users\\maith\\Downloads\\TextMining\\Text_Mining_final\\SVM\\test_predictions.csv', index=False)
print("Predictions saved to 'test_predictions.csv'!")

