import pandas as pd
import re
import nltk
import os
import swifter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from time import time

# Tải dữ liệu cần thiết từ NLTK (nếu chưa có)
## nltk.download('stopwords')
## nltk.download('wordnet')
## nltk.download('omw-1.4')
## nltk.download('punkt')

stop_words = set(stopwords.words('english'))
important_words = {"not", "no", "only", "over"}  # Giữ lại các từ quan trọng
stop_words -= important_words  # Loại bỏ các từ quan trọng khỏi stopwords
lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()

# Chuẩn hóa từ viết tắt
contractions = {
    "can't": "cannot", "won't": "will not", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
}
def expand_contractions(text):
    for key, value in contractions.items():
        text = text.replace(key, value)
    return text

# Hàm làm sạch dữ liệu
def clean_text(review):
    if pd.isna(review) or not isinstance(review, str):
        return ""
    review = review.lower()  # Chuyển về chữ thường
    review = expand_contractions(review)  # Chuẩn hóa từ viết tắt
    review = re.sub(r'http\S+|www\S+', '', review)  # Xóa URL
    review = re.sub(r'\d+', '', review)  # Xóa số
    review = re.sub(r'[^a-z\s]', '', review)  # Loại bỏ ký tự đặc biệt, chỉ giữ lại chữ cái và khoảng trắng
    tokens = tokenizer.tokenize(review)  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Loại bỏ stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Danh sách các file cần xử lý
files = ["train.csv", "test.csv", "validation.csv"]

for file in files:
    if not os.path.exists(file):
        print(f"❌ Error: File '{file}' not found!")
        continue  # Bỏ qua file không tồn tại

    start_time = time()
    print(f"🔄 Processing '{file}'...")

    # Đọc file
    df = pd.read_csv(file)

    # Kiểm tra nếu có cột 'Review'
    if "reviews" not in df.columns:
        print(f"⚠ Warning: 'reviews' column not found in '{file}', skipping...")
        continue

    # Làm sạch dữ liệu
    df["reviews"] = df["reviews"].astype(str).swifter.apply(clean_text)

    # Đổi tên file đầu ra thành dạng "train_clean.csv", "test_clean.csv", "validation_clean.csv"
    output_file = file.replace(".csv", "_clean.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Done! File cleaned and saved to '{output_file}' in {time() - start_time:.2f} seconds.")
