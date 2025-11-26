# ============================================================================
# chroma manager 
# ============================================================================

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
    
    def rebuild_from_posts(self):
        try:
            posts = load_posts()
            if not posts:
                print("[INFO] No posts to add to ChromaDB")
                return
            
            print(f"[INFO] Rebuilding ChromaDB with {len(posts)} posts...")
            
            ids = []
            embeddings = []
            metadatas = []
            
            for post in posts:
                post_id = post.get('post_id')
                if not post_id:
                    continue
                
                try:
                    embedding = post.get('embedding')
                    if not embedding:
                        continue
                    
                    ids.append(f"post_{post_id}")
                    embeddings.append(embedding)
                    metadatas.append({
                        'num_images': len(post.get('images', [])),
                        'post_id': post_id
                    })
                    
                except Exception as e:
                    print(f"[WARN] Failed to process post {post_id}: {e}")
                    continue
            
            if ids:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                print(f"[INFO] Successfully added {len(ids)} posts to ChromaDB")
            else:
                print("[WARN] No valid posts found to add to ChromaDB")
                
        except Exception as e:
            print(f"[ERROR] Failed to rebuild ChromaDB: {e}")
    
    def add_post(self, post_id: int, embedding: np.ndarray, metadata: dict = None):
        try:
            self.delete_post(post_id)
            
            if metadata is None:
                metadata = {}
            
            metadata['post_id'] = post_id
            
            self.collection.add(
                ids=[f"post_{post_id}"],
                embeddings=[embedding.tolist()],
                metadatas=[metadata]
            )
            print(f"[INFO] Added post {post_id} to ChromaDB")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to add post {post_id} to ChromaDB: {e}")
            return False
    
    def delete_post(self, post_id: int):
        try:
            existing = self.collection.get(ids=[f"post_{post_id}"])
            if existing['ids']:
                self.collection.delete(ids=[f"post_{post_id}"])
                print(f"[INFO] Deleted post {post_id} from ChromaDB")
            else:
                print(f"[INFO] Post {post_id} not found in ChromaDB, skipping delete")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to delete post {post_id} from ChromaDB: {e}")
            return False
    
    def query_similar(self, query_embedding: np.ndarray, n_results: int = 100):
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["distances", "metadatas"]
            )
            return results
        except Exception as e:
            print(f"[ERROR] ChromaDB query failed: {e}")
            return None
    
    def get_count(self) -> int:
        try:
            return self.collection.count()
        except:
            return 0