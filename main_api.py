# ============================================================================
# Main 
# ============================================================================

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# 💡 تم تعديل الاستيراد لإضافة Form
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np
import json
from typing import List
import shutil
import uuid
from pydantic import BaseModel 
import requests 

from chroma_manager import ChromaManager
from face_model import images_to_embedding_list, get_face_embedding_from_image
from utils import load_posts, save_posts, cosine_similarity
from config import KNOWN_DIR, SIMILARITY_THRESHOLD, MAX_IMAGES

# ============================================================================
# Flutter Integration Notes:
# 
# Base URL: http://localhost:8000
# 
# Required Endpoints for Flutter:
# 1. POST /posts - Add new missing person
# 2. GET /posts - Get all missing persons 
# 3. DELETE /posts/{id} - Delete missing person
# 4. POST /match - Search for similar persons (تم تغيير الاسم)
# 5. GET /matches/{id} - Get matches for specific person
# 
# Request/Response formats are compatible with Flutter http package
# ============================================================================

app = FastAPI(title="Missing Persons Finder API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

chroma_manager = ChromaManager()

# ============================================================================
# Request Models
# ============================================================================
class SearchRequest(BaseModel):
    # يُستخدم هذا النموذج الآن لإنشاء Post (عند الإرسال من Laravel) وللبحث (/match)
    report_id: int
    image_urls: List[str]

# ============================================================================
# Web Interface Endpoint (Optional - for testing)
# ============================================================================
@app.get("/")
async def root():
    return {"message": "Missing Persons Finder API", "status": "running"}

@app.get("/web")
async def serve_web_interface():
    return FileResponse("index.html")

# ============================================================================
# Core API Endpoints for Flutter App
# ============================================================================

@app.get("/posts")
async def get_all_posts():
    
    try:
        posts = load_posts()
        
        for post in posts:
            if 'images' in post:
                fixed_images = []
                for img_path in post['images']:
                    if 'posts/' in img_path:
                        path_parts = img_path.split('/')
                        if len(path_parts) >= 2:
                            post_id = path_parts[-2]  
                            image_name = path_parts[-1]
                            fixed_images.append(f"/images/{post_id}/{image_name}")
                        else:
                            fixed_images.append(img_path)
                    else:
                        img_name = Path(img_path).name
                        post_id = post['post_id']
                        fixed_images.append(f"/images/{post_id}/{img_name}")
                post['images'] = fixed_images
        
        return {
            "total": len(posts),
            "posts": sorted(posts, key=lambda x: x.get("post_id", 0), reverse=True)
        }
        
    except Exception as e:
        return {
            "total": 0,
            "posts": []
        }

@app.get("/posts/{post_id}")
async def get_post(post_id: int):
    
    posts = load_posts()
    post = next((p for p in posts if p['post_id'] == post_id), None)
    
    if not post:
        raise HTTPException(status_code=404, detail=f"Post ID {post_id} not found")
    
    if 'images' in post:
        fixed_images = []
        for img_path in post['images']:
            if 'posts/' in img_path:
                path_parts = img_path.split('/')
                if len(path_parts) >= 2:
                    post_id = path_parts[-2]
                    image_name = path_parts[-1]
                    fixed_images.append(f"/images/{post_id}/{image_name}")
                else:
                    fixed_images.append(img_path)
            else:
                img_name = Path(img_path).name
                fixed_images.append(f"/images/{post_id}/{img_name}")
        post['images'] = fixed_images
    
    return post

@app.get("/images/{post_id}/{image_name}")
async def get_image(post_id: int, image_name: str):

    image_path = KNOWN_DIR / str(post_id) / image_name
    
    if image_path.exists():
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.post("/posts")
# 💡 التعديل الرئيسي: استقبال JSON (SearchRequest) بدلاً من الملفات المرفوعة
async def create_post(request_data: SearchRequest):
    
    post_id = request_data.report_id
    image_urls = request_data.image_urls

    if len(image_urls) == 0 or len(image_urls) > MAX_IMAGES:
        raise HTTPException(
            status_code=400, 
            detail=f"Please provide 1 to {MAX_IMAGES} image URLs"
        )
    
    posts = load_posts()
    
    if any(p["post_id"] == post_id for p in posts):
        raise HTTPException(
            status_code=400, 
            detail=f"Post ID {post_id} already exists"
        )

    # 💡 منطق جديد: تحميل الصور من الروابط ومعالجة Embeddings مباشرة
    embeddings = []
    
    for url in image_urls:
        try:
            response = requests.get(url, timeout=10) 
            
            if response.status_code != 200:
                print(f"Failed to fetch image from URL: {url} (Status: {response.status_code})")
                continue 
            
            image_data = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Failed to decode image from URL: {url}")
                continue

            emb = get_face_embedding_from_image(img)
            
            if emb is not None:
                embeddings.append(emb)
        except requests.exceptions.RequestException as req_e:
            print(f"Error fetching image from URL {url}: {req_e}")
            continue 

    if not embeddings:
        raise HTTPException(status_code=400, detail="No faces detected in the provided images or images are inaccessible")
    
    # 💡 منطق جديد: حفظ الصور محليًا (اختياريًا، ولكن ضروري لعرضها لاحقًا)
    post_folder = KNOWN_DIR / str(post_id)
    post_folder.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    main_embedding = embeddings[0] # نستخدم أول embedding كـ embedding رئيسي للبوست
    
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                ext = Path(url).suffix or '.jpg'
                image_filename = f"{uuid.uuid4()}{ext}"
                dst = post_folder / image_filename
                
                with open(dst, "wb") as f:
                    f.write(response.content)
                saved_paths.append(str(dst))
        except Exception:
            continue

    
    try:
        new_post = {
            "post_id": post_id,
            "images": saved_paths,
            "embedding": main_embedding.tolist(),
            "matches": []
        }
        
        posts = [p for p in posts if p["post_id"] != post_id]
        posts.append(new_post)
        
        chroma_manager.delete_post(post_id)
        chroma_manager.add_post(
            post_id=post_id,
            embedding=main_embedding,
            metadata={'num_images': len(saved_paths)}
        )
        
        # لا نحتاج لإعادة حساب جميع التطابقات هنا، يمكننا الاعتماد على /match أو queue
        # _recompute_all_matches(posts) 
        save_posts(posts)
        
        return {
            "message": f"Post {post_id} created successfully from URLs",
            "post_id": post_id,
            "images_saved": len(saved_paths)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create post: {str(e)}")
    
    finally:
        # لا حاجة لحذف ملفات مؤقتة لأننا استخدمنا الروابط مباشرة
        pass

@app.delete("/posts/{post_id}")
async def delete_post(post_id: int):

    posts = load_posts()
    post = next((p for p in posts if p['post_id'] == post_id), None)
    
    if not post:
        raise HTTPException(status_code=404, detail=f"Post ID {post_id} not found")
    
    posts = [p for p in posts if p["post_id"] != post_id]
    
    chroma_manager.delete_post(post_id)
    
    folder = KNOWN_DIR / str(post_id)
    if folder.exists():
        try:
            shutil.rmtree(folder)
        except Exception as e:
            print(f"Failed to remove folder: {e}")
    
    _recompute_all_matches(posts)
    save_posts(posts)
    
    return {"message": f"Post {post_id} deleted successfully"}

@app.post("/match")
async def match_report(request_data: SearchRequest):

    report_id = request_data.report_id
    image_urls = request_data.image_urls

    if len(image_urls) == 0 or len(image_urls) > MAX_IMAGES:
        raise HTTPException(
            status_code=400, 
            detail=f"Please provide 1 to {MAX_IMAGES} image URLs"
        )
    
    embeddings = []
    
    try:
        # ⬇️ الخطوة 1: تحميل الصور من الروابط (URLs)
        for url in image_urls:
            try:
                response = requests.get(url, timeout=10) 
                
                if response.status_code != 200:
                    print(f"Failed to fetch image from URL: {url} (Status: {response.status_code})")
                    continue 
                
                # تحويل البيانات الثنائية إلى مصفوفة NumPy ثم إلى صورة لـ OpenCV
                image_data = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"Failed to decode image from URL: {url}")
                    continue

                # ⬇️ الخطوة 2: استخلاص الـ Embedding
                emb = get_face_embedding_from_image(img)
                
                if emb is not None:
                    embeddings.append(emb)
            except requests.exceptions.RequestException as req_e:
                print(f"Error fetching image from URL {url}: {req_e}")
                continue 

        if not embeddings:
            raise HTTPException(status_code=400, detail="No faces detected in the provided images or images are inaccessible")
        
        posts = load_posts()
        if not posts:
            return {
                "match_found": False,
                "similarity": 0.0,
                "matched_report_id": None,
                "report_id": report_id
            }
        
        # ⬇️ الخطوة 3: البحث عن أفضل تطابق
        
        best_match_id = None
        best_similarity = 0.0
        
        for post in posts:
            if post.get('post_id') == report_id: 
                continue

            try:
                post_emb = np.array(post['embedding'], dtype=np.float32)
                
                for search_emb in embeddings:
                    sim = cosine_similarity(search_emb, post_emb)
                    
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match_id = post['post_id']
                        
            except Exception as e:
                continue 

        # ⬇️ الخطوة 4: إرجاع النتيجة لـ Laravel
        if best_similarity >= SIMILARITY_THRESHOLD:
            return {
                "match_found": True,
                "similarity": round(float(best_similarity), 4),
                "matched_report_id": best_match_id, 
                "report_id": report_id
            }
        else:
            return {
                "match_found": False,
                "similarity": round(float(best_similarity), 4),
                "matched_report_id": best_match_id,
                "report_id": report_id
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Matching failed for Report ID {report_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")
    
@app.post("/search")
async def search_similar_posts_by_url(request_data: SearchRequest):
    """
    نقطة نهاية للبحث العام عن التقارير المتطابقة.
    تستقبل: report_id (للسياق) و image_urls.
    تُرجع: قائمة بجميع التقارير المتطابقة التي تتجاوز عتبة التشابه.
    """
    report_id = request_data.report_id
    image_urls = request_data.image_urls

    if not (1 <= len(image_urls) <= MAX_IMAGES):
        raise HTTPException(
            status_code=400, 
            detail=f"الرجاء تقديم من 1 إلى {MAX_IMAGES} روابط صور للبحث"
        )
    
    embeddings = []
    
    # 1. تحميل الصور من الروابط واستخلاص الـ Embedding
    for url in image_urls:
        try:
            response = requests.get(url, timeout=10) 
            if response.status_code != 200:
                print(f"Failed to fetch image from URL: {url} (Status: {response.status_code})")
                continue 
            
            image_data = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Failed to decode image from URL: {url}")
                continue

            emb = get_face_embedding_from_image(img)
            
            if emb is not None:
                embeddings.append(emb)
        except requests.exceptions.RequestException as req_e:
            print(f"Error fetching image from URL {url}: {req_e}")
            continue 

    if not embeddings:
        raise HTTPException(status_code=400, detail="لم يتم اكتشاف أي وجوه في الصور المرفوعة أو الروابط غير صالحة")
    
    try:
        posts = load_posts()
        if not posts:
            return {"search_results": [], "total_matches": 0, "similarity_threshold": SIMILARITY_THRESHOLD}
        
        results = []
        
        # 2. حساب التشابه
        for post in posts:
            try:
                # لا تتم مقارنة التقرير بنفسه
                if post.get('post_id') == report_id: 
                    continue
                    
                post_emb = np.array(post['embedding'], dtype=np.float32)
                
                best_sim = 0.0
                best_img_idx = 0
                
                for i, search_emb in enumerate(embeddings):
                    sim = cosine_similarity(search_emb, post_emb)
                    if sim > best_sim:
                        best_sim = sim
                        # نستخدم فهرس الصورة + 1
                        best_img_idx = i + 1 
                
                if best_sim >= SIMILARITY_THRESHOLD:
                    # إصلاح مسارات الصور قبل الإرجاع
                    fixed_images = []
                    for img_path in post.get('images', []):
                        post_id_from_post = post['post_id']
                        img_name = Path(img_path).name
                        fixed_images.append(f"/images/{post_id_from_post}/{img_name}")
                        
                    post_copy = post.copy()
                    post_copy['images'] = fixed_images
                    
                    results.append({
                        'post': post_copy,
                        'similarity': round(float(best_sim), 4),
                        # نرجع الفهرس ليعرف الكنترولر أي صورة من صور البحث هي الأفضل
                        'best_search_image_index': best_img_idx 
                    })
            except Exception as e:
                print(f"Error processing post ID {post.get('post_id')}: {e}")
                continue
        
        # 3. فرز النتائج
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            "search_results": results,
            "total_matches": len(results),
            "similarity_threshold": SIMILARITY_THRESHOLD
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/matches/{post_id}")
async def get_post_matches(post_id: int):

    posts = load_posts()
    post = next((p for p in posts if p['post_id'] == post_id), None)
    
    if not post:
        raise HTTPException(status_code=404, detail=f"Post ID {post_id} not found")
    
    matches_with_images = []
    for match in post.get("matches", []):
        match_post_id = match.get("post_id")
        match_post = next((p for p in posts if p['post_id'] == match_post_id), None)
        
        if match_post and 'images' in match_post:
            fixed_images = []
            for img_path in match_post['images']:
                if 'posts/' in img_path:
                    path_parts = img_path.split('/')
                    post_id = path_parts[-2]
                    image_name = path_parts[-1]
                    fixed_images.append(f"/images/{post_id}/{image_name}")
                else:
                    img_name = Path(img_path).name
                    fixed_images.append(f"/images/{match_post_id}/{img_name}")
            
            match_with_images = match.copy()
            match_with_images['images'] = fixed_images
            matches_with_images.append(match_with_images)
        else:
            matches_with_images.append(match)
    
    return {
        "post_id": post_id,
        "matches": matches_with_images
    }

@app.post("/recompute-matches")
async def recompute_matches():
    
    posts = load_posts()
    _recompute_all_matches(posts)
    save_posts(posts)
    
    return {"message": "All matches recomputed successfully"}

# ============================================================================
# Debug Endpoints (Optional - for testing)
# ============================================================================

@app.get("/debug/chroma")
async def debug_chroma():
    try:
        count = chroma_manager.get_count()
        posts_count = len(load_posts())
        
        chroma_ids = []
        if count > 0:
            try:
                all_data = chroma_manager.collection.get()
                chroma_ids = all_data['ids']
            except:
                pass
        
        return {
            "chroma_count": count,
            "posts_count": posts_count,
            "status": "healthy" if count == posts_count else "inconsistent",
            "chroma_ids": chroma_ids
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/posts")
async def debug_posts():
    posts = load_posts()
    return {
        "total_posts": len(posts),
        "posts": posts
    }

# ============================================================================
# Internal Functions
# ============================================================================

def _recompute_all_matches(posts):
    if not posts:
        return
    
    if chroma_manager is not None and chroma_manager.get_count() > 0:
        post_dict = {p['post_id']: p for p in posts}
        
        for p1 in posts:
            try:
                emb1 = np.array(p1["embedding"], dtype=np.float32)
                total_posts = chroma_manager.get_count()
                n_results = min(100, total_posts)
                
                results = chroma_manager.query_similar(emb1, n_results=n_results)
                matches = []
                
                if results and results.get('ids') and len(results['ids'][0]) > 0:
                    for i, post_id_str in enumerate(results['ids'][0]):
                        try:
                            post_id = int(post_id_str.split('_')[1])
                            
                            if post_id == p1['post_id']:
                                continue
                            
                            if post_id not in post_dict:
                                continue
                                
                            emb2 = np.array(post_dict[post_id]['embedding'], dtype=np.float32)
                            similarity = cosine_similarity(emb1, emb2)
                            
                            if similarity >= SIMILARITY_THRESHOLD:
                                matches.append({
                                    "post_id": post_id,
                                    "similarity": round(float(similarity), 4)
                                })
                        except Exception as e:
                            continue
                
                p1["matches"] = sorted(matches, key=lambda x: x["similarity"], reverse=True)
                
            except Exception as e:
                p1["matches"] = []
    else:
        for i, p1 in enumerate(posts):
            matches = []
            try:
                emb1 = np.array(p1["embedding"], dtype=np.float32)
            except Exception:
                p1["matches"] = []
                continue
            
            for j, p2 in enumerate(posts):
                if i == j:
                    continue
                
                try:
                    emb2 = np.array(p2["embedding"], dtype=np.float32)
                except Exception:
                    continue
                
                sim = cosine_similarity(emb1, emb2)
                if sim >= SIMILARITY_THRESHOLD:
                    matches.append({
                        "post_id": p2["post_id"],
                        "similarity": round(float(sim), 4)
                    })
            
            p1["matches"] = sorted(matches, key=lambda x: x["similarity"], reverse=True)


if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)