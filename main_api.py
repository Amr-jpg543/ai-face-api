# ============================================================================

# main_api.py - FastAPI AI Engine (Laravel Compatible & Flexible)

# ============================================================================



from pathlib import Path

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

from typing import List, Optional

import uvicorn

import cv2

import numpy as np

import requests

import uuid

import shutil



from chroma_manager import ChromaManager

from face_model import get_face_embedding_from_image

from config import KNOWN_DIR, SIMILARITY_THRESHOLD, MAX_IMAGES

from utils import create_post, delete_post, get_post_by_id



# ============================================================================

# FastAPI Setup

# ============================================================================



app = FastAPI(title="Missing Persons AI API", version="1.0")



chroma_manager = ChromaManager()



# ============================================================================

# Root

# ============================================================================



@app.get("/")

def root():

    return {"status": "AI Engine Running"}



# ============================================================================

# CREATE POST (Laravel sends report)

# ============================================================================



@app.post("/posts")

async def create_post_endpoint(body: dict):



    # ✅ Accept either 'image_urls_json' or 'image_urls'

    post_id = body.get("report_id")

    image_urls = body.get("image_urls_json") or body.get("image_urls")



    if post_id is None or not image_urls:

        raise HTTPException(400, "report_id and image_urls_json (or image_urls) are required")



    if len(image_urls) > MAX_IMAGES:

        raise HTTPException(400, f"Send 1 to {MAX_IMAGES} images")



    embeddings = []

    saved_paths = []



    post_folder = KNOWN_DIR / str(post_id)

    post_folder.mkdir(parents=True, exist_ok=True)



    for url in image_urls:

        try:

            r = requests.get(url, timeout=10)

            img_data = np.frombuffer(r.content, np.uint8)

            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)



            emb = get_face_embedding_from_image(img)

            if emb is not None:

                embeddings.append(emb)



            # Save image locally

            ext = Path(url).suffix or ".jpg"

            filename = f"{uuid.uuid4()}{ext}"

            dst = post_folder / filename

            with open(dst, "wb") as f:

                f.write(r.content)

            saved_paths.append(str(dst))



        except Exception as e:

            print("[WARN] Image download/processing failed:", e)



    if not embeddings:

        raise HTTPException(400, "No faces detected in images")



    # Average embedding

    avg_embedding = np.mean(embeddings, axis=0)



    # Save to JSON DB

    create_post(post_id, saved_paths, avg_embedding.tolist())



    # Save to Chroma

    chroma_manager.add_post(post_id, avg_embedding, {"num_images": len(saved_paths)})



    return {

        "success": True,

        "message": "Post created",

        "post_id": post_id,

        "images_saved": len(saved_paths)

    }



# ============================================================================

# DELETE POST

# ============================================================================



@app.delete("/posts/{post_id}")

async def delete_post_endpoint(post_id: int):



    post = get_post_by_id(post_id)

    if not post:

        raise HTTPException(404, "Post not found")



    delete_post(post_id)

    chroma_manager.delete_post(post_id)



    folder = KNOWN_DIR / str(post_id)

    if folder.exists():

        shutil.rmtree(folder)



    return {"message": f"Post {post_id} deleted"}



# ============================================================================

# SEARCH MATCHES

# ============================================================================



@app.post("/search")

async def search_faces(body: dict):



    # Accept both keys

    image_urls = body.get("image_urls_json") or body.get("image_urls")

    if not image_urls:

        return {"search_results": []}



    embeddings = []

    for url in image_urls:

        try:

            r = requests.get(url, timeout=10)

            img_data = np.frombuffer(r.content, np.uint8)

            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)



            emb = get_face_embedding_from_image(img)

            if emb is not None:

                embeddings.append(emb)



        except Exception:

            continue



    if not embeddings:

        return {"search_results": []}



    # Average search embedding

    search_emb = np.mean(embeddings, axis=0)



    # Search Chroma

    results = chroma_manager.search(search_emb, top_k=5)



    formatted = []

    for r in results:

        formatted.append({

            "post": {"post_id": r["post_id"]},

            "similarity": float(r["similarity"]),

            "best_search_image_index": 0

        })



    return {"search_results": formatted}



# ============================================================================

# Run

# ============================================================================



if __name__ == "__main__":

    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)