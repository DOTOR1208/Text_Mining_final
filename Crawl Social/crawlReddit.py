import praw
import csv
from langdetect import detect
from transformers import pipeline

# Initialize sentiment analysis pipeline from Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Create a Reddit API instance
reddit = praw.Reddit(client_id='your_client_id',
                     client_secret='your_client_secret',
                     user_agent='thuong (by /u/SquirrelPlus4874)')

# List of Reddit post URLs (replace these with your actual URLs)
reddit_urls = [
    "https://www.reddit.com/r/Wellthatsucks/comments/1em72lc/dog_chews_on_liion_battery_causing_house_fire/",
    "https://www.reddit.com/r/Damnthatsinteresting/comments/1ai0run/firefighter_putting_out_a_fire_using_bernoullis/",
    "https://www.reddit.com/r/travel/comments/13hmnlx/what_i_like_and_dislike_about_the_usa_as_a_tourist/",
    "https://www.reddit.com/r/interestingasfuck/comments/1hmo09d/china_has_just_unveiled_a_new_heavy_stealth/",
    "https://www.reddit.com/r/interestingasfuck/comments/188ieq7/musket_vs_medieval_armor_in_slow_motion/",
    "https://www.reddit.com/r/interestingasfuck/comments/1i06qx1/these_men_are_not_special_forces_they_are_members/",
    "https://www.reddit.com/r/interestingasfuck/comments/1i00ltp/california_has_incarcerated_firefighters/",
    "https://www.reddit.com/r/interestingasfuck/comments/1hzkxse/one_guy_changed_the_entire_outcome_of_this_video/",
    "https://www.reddit.com/r/interestingasfuck/comments/1hzihpt/thai_mens_national_team_meets_taiwan_womens/",
    "https://www.reddit.com/r/interestingasfuck/comments/1hzkh0l/this_is_a_tsunami_escape_pod/",
    "https://www.reddit.com/r/interestingasfuck/comments/1hz8ps4/women_submerged_five_sets_of_her_fine_china/",
    "https://www.reddit.com/r/NoStupidQuestions/comments/13jl1hy/what_is_the_closest_i_can_get_to_an_unbiased_news/",
    "https://www.reddit.com/r/aliens/comments/1hdwtd1/are_we_in_disclosure_abc_news_aired_30_seconds_of/",
    "https://www.reddit.com/r/worldnews/comments/1hpz2iw/bbc_news_us_treasury_says_it_was_hacked_by_china/",
    "https://www.reddit.com/r/economicCollapse/comments/1h667e5/todays_unsurprising_news/",
    "https://www.reddit.com/r/Damnthatsinteresting/comments/1gj6uxv/volkswagens_new_emergency_assist_technology/",
    "https://www.reddit.com/r/pcmasterrace/comments/1hdnjo6/ray_tracing_is_an_innovative_technology_bro_its/",
    "https://www.reddit.com/r/interestingasfuck/comments/1gj1a5p/when_willpower_combined_with_technology_can_take/"
    # Add more URLs as needed
]

# Open the CSV file in write mode
with open('Reddit_Comments.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Comment', 'Sentiment', 'Source'])

    # Counter to track the number of comments processed
    comment_count = 0
    max_comments = 5000

    # Loop through each Reddit post URL
    for url in reddit_urls:
        print(f"Processing URL: {url}")  # Debugging statement
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=None)  # To avoid the 'load more comments' button

        # Iterate through the comments
        for comment in submission.comments:
            # Check if we've reached the max comment limit
            if comment_count >= max_comments:
                print(f"Max comment limit reached: {max_comments}")  # Debugging statement
                break

            # Language detection and sentiment analysis
            try:
                if detect(comment.body) == 'en':
                    sentiment_result = sentiment_analyzer(comment.body[:512])[0]  # Truncate to 512 tokens
                    writer.writerow([comment.body, sentiment_result['label'], 'Reddit'])
                    comment_count += 1
            except Exception as e:
                print(f"Error processing comment: {e}")
                continue

        # If we've reached the limit, break out of the outer loop as well
        if comment_count >= max_comments:
            break

    print(f"Comments saved to 'Reddit_Comments.csv' (Total {comment_count} comments).")
