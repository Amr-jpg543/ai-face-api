# ============================================================================
# Utility Functions - Local JSON Storage
# ============================================================================

import json
import numpy as np
from config import POSTS_JSON


# ==============================
# Load all posts
# ==============================
def load_posts():
    if not POSTS_JSON.exists():
        return []
    try:
        with open(POSTS_JSON, "r") as f:
            return json.load(f)
    except Exception:
        return []


# Alias for old model compatibility
def get_all_posts():
    return load_posts()


# ==============================
# Save all posts
# ==============================
def save_posts(posts):
    with open(POSTS_JSON, "w") as f:
        json.dump(posts, f, indent=2)


# ==============================
# Create / Update Post
# ==============================
def create_post(post_id: int, images: list, embedding: list):
    posts = load_posts()

    # Remove old post if exists
    posts = [p for p in posts if p["post_id"] != post_id]

    posts.append({
        "post_id": post_id,
        "images": images,
        "embedding": embedding
    })

    save_posts(posts)


# ==============================
# Get Post by ID
# ==============================
def get_post_by_id(post_id: int):
    posts = load_posts()
    for p in posts:
        if p["post_id"] == post_id:
            return p
    return None


# ==============================
# Delete Post
# ==============================
def delete_post(post_id: int):
    posts = load_posts()
    posts = [p for p in posts if p["post_id"] != post_id]
    save_posts(posts)


# ==============================
# Cosine Similarity (used by face_model)
# ==============================
def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))
