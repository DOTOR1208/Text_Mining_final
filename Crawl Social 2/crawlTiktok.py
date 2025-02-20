import requests
import csv
import json
import re
import emoji
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import pipeline

# Khởi tạo mô hình phân tích cảm xúc
sentiment_analyzer = pipeline("sentiment-analysis")

post_urls = [
    'https://www.tiktok.com/@learnenglishonline_6/video/7433779167948197152',
    'https://www.tiktok.com/@evolvedata22/video/7250355072511626539',
    'https://www.tiktok.com/@mrbeast/video/7452741134217923886'
]

output_file = 'Tiktok_Comments_Sentiment.csv'

headers = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'no-cache',
    'pragma': 'no-cache',
    'priority': 'u=1, i',
    'referer': 'https://www.tiktok.com/',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
}

def req(post_id, cursor):
    """Gửi request lấy dữ liệu bình luận từ TikTok"""
    url = f'https://www.tiktok.com/api/comment/list/?aid=1988&count=20&cursor={cursor}&aweme_id={post_id}'
    response = requests.get(url=url, headers=headers)
    
    if response.status_code == 200 and response.text.strip():
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON.")
            return None
    else:
        print(f"Error: Invalid response (Status Code: {response.status_code})")
        return None

def remove_emoji(text):
    """Loại bỏ emoji khỏi văn bản"""
    text = emoji.replace_emoji(text, replace="")  # Thay emoji bằng chuỗi rỗng
    text = re.sub(r'[^\w\s,!?\'\".-]', '', text)  # Xóa ký tự đặc biệt không mong muốn
    return text.strip()

def analyze_sentiment(comment):
    """Phân tích cảm xúc của bình luận"""
    result = sentiment_analyzer(comment)
    label = result[0]['label']
    return 'positive' if label == 'POSITIVE' else 'negative'

def parser(data):
    unique_comments = set()  # Dùng set để tránh trùng lặp
    comments = []

    for cm in data.get('comments', []):
        com = cm['text'] if cm['text'] else cm['share_info']['desc'] if cm['share_info']['desc'] else ''
        if com:
            try:
                com = remove_emoji(com)  # Xóa emoji
                com = com.strip()  # Xóa khoảng trắng đầu/cuối
                if len(com) >= 10 and detect(com) == 'en' and com not in unique_comments:  # Lọc theo độ dài + tiếng Anh + không trùng
                    sentiment = analyze_sentiment(com)  # Phân tích cảm xúc
                    unique_comments.add(com)
                    comments.append([com, sentiment, 'Tiktok'])
            except LangDetectException:
                pass  # Bỏ qua nếu không phát hiện được ngôn ngữ
    return comments

def save_to_csv(comments, writer, post_id):
    writer.writerow([f'Comments for video: {post_id}', '', ''])
    writer.writerows(comments)

# Crawl dữ liệu
with open(output_file, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['text', 'label', 'source'])  # Tiêu đề cột

    for post_url in post_urls:
        post_id = post_url.split('/')[-1]
        print(f"Processing video: {post_url} (Post ID: {post_id})")
        
        comments = []
        cursor = 0

        while True:
            raw_data = req(post_id, cursor)
            if raw_data is None:
                break
            
            comments += parser(raw_data)

            if raw_data.get('has_more') == 1:
                cursor = raw_data.get('cursor', cursor + 100)
                print(f"Moving to next cursor: {cursor}")
            else:
                break
        
        save_to_csv(comments, writer, post_id)

print(f"Data saved to {output_file}")