import pandas as pd
import re
import nltk
import os
import swifter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from time import time

# Tải dữ liệu cần thiết từ NLTK (chỉ cần chạy 1 lần nếu chưa tải)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

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

# Đọc dữ liệu
input_file = "data_13_26.csv"
output_file = "clean_data_13_26.csv"

if not os.path.exists(input_file):
    print(f"❌ Error: File '{input_file}' not found!")
else:
    start_time = time()
    df = pd.read_csv(input_file, header=None, names=["ID", "Review", "Rating"])
    df["Review"] = df["Review"].astype(str).swifter.apply(clean_text)  # Dùng swifter để tăng tốc
    df.to_csv(output_file, index=False, header=False)
    print(f"✅ Done! File cleaned and saved to {output_file} in {time() - start_time:.2f} seconds.")
