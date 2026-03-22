# chroma_manager.py

import chromadb
import numpy as np
from config import SIMILARITY_THRESHOLD
from utils import load_posts

class ChromaManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="face_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

        if self.get_count() == 0:
            self.rebuild_from_posts()

    # ==============================
    # Load all posts embeddings
    # ==============================
    def rebuild_from_posts(self):
        posts = load_posts()
        if not posts:
            print("[INFO] No posts found")
            return

        ids, embeddings, metadatas = [], [], []

        for post in posts:
            post_id = post.get("post_id")
            embedding = post.get("embedding")

            if not post_id or embedding is None:
                continue

            ids.append(f"post_{post_id}")
            embeddings.append(embedding)
            metadatas.append({
                "post_id": post_id,
                "num_images": len(post.get("images", []))
            })

        if ids:
            self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
            print(f"[INFO] Loaded {len(ids)} embeddings into Chroma")

    # ==============================
    # Add / Delete
    # ==============================
    def add_post(self, post_id: int, embedding: np.ndarray, metadata: dict = None):
        self.delete_post(post_id)

        metadata = metadata or {}
        metadata["post_id"] = post_id

        self.collection.add(
            ids=[f"post_{post_id}"],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

        print(f"[INFO] Post {post_id} added to Chroma")

    def delete_post(self, post_id: int):
        try:
            self.collection.delete(ids=[f"post_{post_id}"])
        except:
            pass

    # ==============================
    # Search function (IMPORTANT)
    # ==============================
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["distances", "metadatas"]
        )

        matches = []

        for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
            similarity = 1 - dist  # 🔥 convert cosine distance → similarity

            if similarity < SIMILARITY_THRESHOLD:
                continue

            matches.append({
                "post_id": meta["post_id"],
                "similarity": float(similarity)
            })

        return matches

    def get_count(self):
        return self.collection.count()
