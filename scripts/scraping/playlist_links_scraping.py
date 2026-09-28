#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

# Allow this script to import the project config when launched by file path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import VIDEO_LINKS_PATH

def extract_urls(source):
    """Extract URLs from a single source."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(webpage_url)s",
        "--no-warnings",
        source
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return [u for u in result.stdout.strip().split('\n') if u]

# Process multiple playlists/channels
sources = [
    "https://www.youtube.com/watch?v=RDd7HD2tn2g&list=PLg7NjHK9t-vvAzmSAt-ptt1Ses3MX6R-R",
    "https://www.youtube.com/watch?v=vxrEAAf7LSQ&list=PLg7NjHK9t-vtLiG4RpMwc_Vquc0T1U15V"
]

all_urls = []
for source in sources:
    print(f"Processing: {source}")
    urls = extract_urls(source)
    all_urls.extend(urls)
    print(f"  Found {len(urls)} videos")

# Remove duplicates and save
unique_urls = list(set(all_urls))
VIDEO_LINKS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(VIDEO_LINKS_PATH, "w", encoding="utf-8") as f:
    f.write('\n'.join(unique_urls))

print(f"\nTotal unique URLs: {len(unique_urls)}")
