import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Đọc dữ liệu train
train_df = pd.read_csv("train_clean.csv")

# 2️⃣ Biến đổi văn bản thành vector TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train_df["reviews"])
y_train = train_df["rating"]

# 3️⃣ Train mô hình Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 🛠 Tạo thư mục nếu chưa có
model_dir = "Decision_Tree"
os.makedirs(model_dir, exist_ok=True)

# 4️⃣ Lưu mô hình và TF-IDF vectorizer
joblib.dump(dt_model, os.path.join(model_dir, "decision_tree_model.pkl"))
joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

print("✅ Model trained and saved!")

# 5️⃣ Đọc dữ liệu test để đánh giá
test_df = pd.read_csv("test_clean.csv")

# 6️⃣ Load mô hình đã train
dt_model = joblib.load(os.path.join(model_dir, "decision_tree_model.pkl"))
tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))

# 7️⃣ Biến đổi dữ liệu test
X_test = tfidf.transform(test_df["reviews"])
y_test = test_df["rating"]

# 8️⃣ Dự đoán trên tập test
y_pred = dt_model.predict(X_test)

# 9️⃣ In kết quả đánh giá
print(f"🔍 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))