import requests
import pandas as pd
import time

#API key IDMbIDMb
API_KEY = "54e6f18fa4628b039afbbb93e4e6a3f5"

#DDanh sách phim phổ biến từ TMDb
def get_popular_movies(page=1):
    url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&page={page}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print(f"Error fetching page {page}: {response.status_code}")
        return []

#ĐĐánh giá của một bộ phim
def get_movie_reviews(movie_id):
    reviews = []
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        review_data = response.json()["results"]
        for review in review_data:
            reviews.append({
                "movie_id": movie_id,
                "author": review["author"],
                "content": review["content"],
                "url": f"https://www.themoviedb.org/review/{review['id']}",
                "source": "TMDb"
            })
    return reviews

# Crawl funcfunc
def crawl_data():
    all_reviews = []
    page = 1
    reviews_collected = 0

    while reviews_collected < 40000:
        print(f"Fetching page {page} of popular movies...")
        movies = get_popular_movies(page)
        if not movies:
            break

        for movie in movies:
            movie_id = movie["id"]
            reviews = get_movie_reviews(movie_id)
            all_reviews.extend(reviews)
            reviews_collected += len(reviews)

            if reviews_collected >= 40000:
                break

        page += 1
        time.sleep(1)  # Thêm độ trễ để tránh bị block IP

    return all_reviews

# Lưu dữ liệu
def save_to_csv(reviews):
    df = pd.DataFrame(reviews)
    df.to_csv("movie_reviews_2.csv", index=False)
    print(f"Data saved to movie_reviews_2.csv with {len(reviews)} reviews.")

# Main function
def main():
    reviews = crawl_data()
    save_to_csv(reviews)

if __name__ == "__main__":
    main()
