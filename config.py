from pathlib import Path

# 👇 أهم تعديل
BASE_DIR = Path("/data")

KNOWN_DIR = BASE_DIR / "posts"
KNOWN_DIR.mkdir(parents=True, exist_ok=True)

POSTS_JSON = BASE_DIR / "posts.json"

SIMILARITY_THRESHOLD = 0.20
OUTLIER_HIGH_THRESHOLD = 0.85
OUTLIER_LOW_THRESHOLD = 0.25

MAX_IMAGES = 5
AUTO_REFRESH_MS = 3000

print(f"[CONFIG] Project directory: {BASE_DIR}")
print(f"[CONFIG] KNOWN_DIR: {KNOWN_DIR}")
print(f"[CONFIG] POSTS_JSON: {POSTS_JSON}")
print(f"[CONFIG] SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}")