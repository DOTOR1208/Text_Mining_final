import os
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report

# 🟢 HÀM LOẠI BỎ TỪ ÍT XUẤT HIỆN
def remove_rare_words(df, min_count=2):
    word_counts = pd.Series(" ".join(df["reviews"]).split()).value_counts()
    rare_words = set(word_counts[word_counts < min_count].index)
    df["reviews"] = df["reviews"].apply(lambda x: " ".join([word for word in x.split() if word not in rare_words]))
    return df

# 1️⃣ Đọc dữ liệu train
train_df = pd.read_csv("train_clean.csv")
train_df = remove_rare_words(train_df, min_count=2)  # Loại bỏ từ ít xuất hiện

# 2️⃣ TF-IDF với tối ưu tham số
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=10000, min_df=5, max_df=0.9)
X_train = tfidf.fit_transform(train_df["reviews"])
y_train = train_df["rating"]

# 3️⃣ Train mô hình Complement Naive Bayes
nb_model = ComplementNB(alpha=0.1)  # Chỉnh alpha để tránh overfitting
nb_model.fit(X_train, y_train)

# 🛠 Tạo thư mục nếu chưa có
model_dir = "Naive_Bayes"
os.makedirs(model_dir, exist_ok=True)

# 4️⃣ Lưu mô hình và vectorizer
joblib.dump(nb_model, os.path.join(model_dir, "naive_bayes_model.pkl"))
joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

print("✅ Model trained and saved!")

# 5️⃣ Đọc dữ liệu test
test_df = pd.read_csv("test_clean.csv")
test_df = remove_rare_words(test_df, min_count=2)

# 6️⃣ Load mô hình đã train
nb_model = joblib.load(os.path.join(model_dir, "naive_bayes_model.pkl"))
tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))

# 7️⃣ Biến đổi dữ liệu test
X_test = tfidf.transform(test_df["reviews"])
y_test = test_df["rating"]

# 8️⃣ Dự đoán trên tập test
y_pred = nb_model.predict(X_test)

# 9️⃣ In kết quả đánh giá
print(f"🔍 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
