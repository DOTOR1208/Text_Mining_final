import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'br', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load dữ liệu train và test
train_df = pd.read_csv('train_clean.csv')  # Thay đổi tên file train thực tế
test_df = pd.read_csv('test_clean.csv')    # Sử dụng file test bạn đã cung cấp

# Tiền xử lý dữ liệu
train_df['reviews_clean'] = train_df['reviews'].apply(preprocess_text)
test_df['reviews_clean'] = test_df['reviews'].apply(preprocess_text)

# Chuẩn bị dữ liệu
X_train = train_df['reviews_clean']
y_train = train_df['rating']
X_test = test_df['reviews_clean']
y_test = test_df['rating']

# Vector hóa TF-IDF (chỉ fit trên train)
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)  # Chỉ transform trên test

# Huấn luyện SVM
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train_tfidf, y_train)

# Dự đoán và đánh giá
y_pred = svm.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Kết quả đánh giá trên tập test:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# Xuất kết quả dự đoán
output_df = test_df[['asins', 'reviews']].copy()
output_df['predicted_rating'] = y_pred
output_df.to_csv('test_predictions.csv', index=False)