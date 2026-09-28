import json
import sys
from pathlib import Path

from youtube_comment_downloader import *

# Allow this script to import the project config when launched by file path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import COMMENTS_JSON_PATH, SCRAPED_LINKS_PATH, VIDEO_LINKS_PATH

downloader = YoutubeCommentDownloader()

# Read video links and already scraped links
with open(VIDEO_LINKS_PATH, "r", encoding="utf-8") as f:
    video_links = f.read().splitlines()

SCRAPED_LINKS_PATH.parent.mkdir(parents=True, exist_ok=True)
SCRAPED_LINKS_PATH.touch(exist_ok=True)
with open(SCRAPED_LINKS_PATH, "r", encoding="utf-8") as f:
    already_scraped_links = set(f.read().splitlines())

# Process each video link
for i,video_link in enumerate(video_links):
    if video_link in already_scraped_links:
        print(f"Skipping {video_link}")
        continue
    else:
        comments = downloader.get_comments_from_url(video_link)
        comments = list(comments) # Convert the generator to a list to avoid exhaustion
        # Save comments to a JSON file
        COMMENTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(COMMENTS_JSON_PATH, "a", encoding="utf-8") as f:
            for comment in comments:
                comment_data = {
                    "text": comment['text'],
                    "video_url": video_link
                }
                f.write(json.dumps(comment_data, ensure_ascii=False) + "\n")
        print(f"Processed {video_link} with {len(comments)} comments.")
        print(f"{i+1}/{len(video_links)} videos processed.")
        with open(SCRAPED_LINKS_PATH, "a", encoding="utf-8") as f:
            f.write(f"{video_link}\n")
