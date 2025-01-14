from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import csv
import logging
from langdetect import detect

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# YouTube Data API Key
api_key = "AIzaSyBjc2O3WUnNrbcml1y2FGM16DXM2bgO4BM"

# Fixed list of video IDs
video_ids = [
    "Oa0ZHfcalCM",
    "5puu3kN9l7c",
    "p9Q5a1Vn-Hk",
    "62wEk02YKs0",
    "eEWa7cpiyD8",
]

output_file = "Youtube_Comments.csv"
MAX_COMMENTS = 5000  # Limit to 5000 comments across all videos

# Build YouTube API client
yt_client = build("youtube", "v3", developerKey=api_key)


def get_comments(client, video_id, token=None):
    try:
        response = client.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,  # Maximum allowed is 100
            pageToken=token,
        ).execute()
        return response
    except HttpError as e:
        logging.error(f"HTTP Error {e.resp.status}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error: {e}")
        return None


comments = []
total_fetched = 0  # Counter to track the number of comments fetched across all videos

logging.info("Fetching comments...")

# Loop through all fixed video IDs
for vid_id in video_ids:
    next_page_token = None
    logging.info(f"Fetching comments for video: {vid_id}")

    while total_fetched < MAX_COMMENTS:
        response = get_comments(yt_client, vid_id, next_page_token)
        if not response:
            break

        for item in response["items"]:
            if total_fetched >= MAX_COMMENTS:
                break

            comment_data = item["snippet"]["topLevelComment"]["snippet"]
            comment_text = comment_data["textDisplay"]

            try:
                # Detect language of the comment
                if detect(comment_text) == 'en':  # Check if comment is in English
                    comments.append([
                        comment_text,   # Comment text
                        "",             # Blank sentiment
                        "Youtube"       # Default source
                    ])
                    total_fetched += 1
            except Exception as e:
                logging.error(f"Error detecting language: {e}")
                continue

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

logging.info(f"Total English comments fetched: {len(comments)}")

# Write to CSV
with open(output_file, "w", newline="", encoding="utf-8") as file:
    csv_writer = csv.writer(file)
    # Write header
    csv_writer.writerow(["Comment", "Sentiment", "Source"])
    csv_writer.writerows(comments)

logging.info(f"Comments saved to {output_file}")
