import requests
import csv
import time

# API Key and Base URL
api_key = "678a1c6b7febe5def8e11493" #Scrapingdog API Key
url = "https://api.scrapingdog.com/amazon/reviews"

# List of ASINs (Amazon Product IDs)
asins = ["B07RZ74VLR", "B0BTKJFRDV", "B09MLRPTT2", "B07T5SY43L", "B087LXCTFJ", "B07GBZ4Q68", "B07CMS5Q6P", "B07BVK2FQW", "B0D2FZS3JM", "B01H6GUCCQ", "B09VV5LJS1", "B086PKMZ21", "B07TXM7K4T", "B07XF2TGGK", "B079CBP6P9", "B07CMS5Q6P", "B07L4BM851", "B07ZGDPT4M", "B081415GCS", "B00Z0UWWYC", "B0CGVY3YYB", "B07QNZC9V5", "B07MRMHML9", "B00NLZUM36", "B07GCKQD77", "B00YXO5U40", "B08NTYB4M7", "B086PKMZ1Q", "B094PS5RZQ", "B07RGHS919", "B07YN82X3B", "B09C13PZX7", "B00ABA0ZOA", "B07QNZC9V5", "B00Z9V0NKC", "B0B8QMW657", "B0B4B2HW2N", "B098LG3N6R", "B07FLKYRFB", "B07Y693ND1", "B07NY9ZT92", "B09GTRVJQM", "B07VVK39F7", "B00H2VOSP8", "B0932M1666", "B06X421WJ6", "B07T7W7VR3", "B0866S3D82", "B093KYLLCG", "B08MW43SRM", "B095YJW56C", "B08PBG6QHZ", "B08GHX9G5L"]  # Add more ASINs of products here

# CSV file to save reviews
csv_filename = "amazon_reviews.csv"

# Target number of reviews
target_reviews = 40000

# Function to check if a review is in English
def is_english(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# Open CSV file for writing
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["asin", "review", "rating"])
    writer.writeheader()

    total_reviews = 0

    # Iterate through each ASIN
    for asin in asins:
        print(f"Fetching reviews for ASIN: {asin}")
        page = 1

        while True:
            # Break if we have collected enough reviews
            if total_reviews >= target_reviews:
                break

            # Set up parameters for the API request
            params = {
                "api_key": api_key,
                "asin": asin,
                "domain": "com",
                "page": str(page)
            }

            # Make the API request with retries
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        break
                    else:
                        print(f"Failed to fetch page {page} for ASIN {asin}. Status code: {response.status_code}. Retrying...")
                        retry_count += 1
                        time.sleep(2)  # Wait before retrying
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}. Retrying...")
                    retry_count += 1
                    time.sleep(2)  # Wait before retrying

            # If all retries failed, move to the next ASIN
            if retry_count == max_retries:
                print(f"Max retries reached for ASIN {asin}. Skipping to the next ASIN.")
                break

            # Parse the JSON response
            data = response.json()

            # Check if there are no more reviews
            if not data.get("customer_reviews"):
                print(f"No more reviews found for ASIN {asin} at page {page}.")
                break

            # Extract reviews and ratings
            for review_data in data.get("customer_reviews", []):
                review = review_data.get("review", "")
                rating = review_data.get("rating", "")

                # Only write English reviews
                if is_english(review):
                    writer.writerow({"asin": asin, "review": review, "rating": rating})
                    total_reviews += 1

                    # Break if we have collected enough reviews
                    if total_reviews >= target_reviews:
                        break

            print(f"Page {page} for ASIN {asin} completed. Total reviews collected: {total_reviews}")

            # Move to the next page
            page += 1

            # Add a delay to avoid hitting rate limits
            time.sleep(1)  # Adjust as needed

        # Break if we have collected enough reviews
        if total_reviews >= target_reviews:
            break

print(f"All reviews have been saved to {csv_filename}. Total reviews collected: {total_reviews}")