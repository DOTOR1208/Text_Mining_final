import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

# Tải dữ liệu cần thiết từ NLTK (chỉ cần chạy 1 lần nếu chưa tải)
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
important_words = {"not", "no", "only", "over"}  # Giữ lại các từ quan trọng
stop_words -= important_words  # Loại bỏ các từ quan trọng khỏi stopwords
lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()

# Hàm pipeline làm sạch dữ liệu
def pipeline(review):
    review = review.lower()  # Chuyển về chữ thường
    review = re.sub(r'[^\w\s]', '', review)  # Loại bỏ dấu câu
    tokens = tokenizer.tokenize(review)  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Loại bỏ stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

input_file = "data_26_40.csv"
df = pd.read_csv(input_file, header=None, names=["ID", "Review", "Rating"])

# Làm sạch dữ liệu trong cột Review
df["Review"] = df["Review"].astype(str).apply(pipeline)

# Lưu file kết quả
output_file = "clean_data_26_40.csv"
df.to_csv(output_file, index=False, header=False)

print(f"✅ Done! File cleaned and save to {output_file}")