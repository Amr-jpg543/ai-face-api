# =====================================================================
# Configuration (Ready for Railway Volume)
# =====================================================================

import os
from pathlib import Path

# لو Railway مركب Volume → استخدمه
VOLUME_DIR = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/data")

BASE_DIR = Path(VOLUME_DIR)

# مكان ملفات الصور/البيانات
KNOWN_DIR = BASE_DIR / "posts"
KNOWN_DIR.mkdir(parents=True, exist_ok=True)

# JSON الأساسي
POSTS_JSON = BASE_DIR / "posts.json"

# إعدادات البحث
SIMILARITY_THRESHOLD = 0.20
OUTLIER_HIGH_THRESHOLD = 0.85
OUTLIER_LOW_THRESHOLD = 0.25

MAX_IMAGES = 5
AUTO_REFRESH_MS = 3000

print(f"[CONFIG] Using BASE_DIR: {BASE_DIR}")
print(f"[CONFIG] KNOWN_DIR: {KNOWN_DIR}")
print(f"[CONFIG] POSTS_JSON: {POSTS_JSON}")
