from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from typing import List
import cv2, numpy as np, base64, os
from insightface.app import FaceAnalysis
from pathlib import Path
import io


app = FastAPI()


# Test Terminal
@app.get("/")
def home():
    return "Hello World"


# Registration Terminal

DB_PATH = "students_db.npz"
DET_SIZE = 320
# Load face analysis model globally (faster than loading inside function)
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))

def l2_normalize(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def load_database(db_path):
    """Loads labels and embeddings from DB if exists, else returns empty."""
    if not os.path.exists(db_path):
        print(f"[INFO] No database found at {db_path}. Creating a new one.")
        return [], np.empty((0, 512), dtype=np.float32)  # Empty DB

    data = np.load(db_path, allow_pickle=True)
    labels = data["labels"].tolist()
    embeddings = data["embeddings"].astype(np.float32)
    return labels, embeddings

def save_database(labels, embeddings, db_path):
    np.savez_compressed(db_path, labels=np.array(labels, dtype=object), embeddings=embeddings.astype(np.float32))

@app.post("/register")
async def register_student(
    enroll_no: str = Form(...),
    images: List[UploadFile] = None
):
    if not images or len(images) != 3:
        return {"error": "Please upload exactly 3 images."}

    embeddings = []
    for img_file in images:
        img_bytes = await img_file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        faces = face_app.get(img)
        if not faces:
            return {"error": f"No face detected in {img_file.filename}"}

        # Take largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embeddings.append(l2_normalize(face.embedding))

    # Compute mean embedding
    mean_emb = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))

    # Load DB
    labels, embeddings_db = load_database(DB_PATH)

    # Check if enroll_no already exists → update
    if enroll_no in labels:
        idx = labels.index(enroll_no)
        embeddings_db[idx] = mean_emb
        msg = f"Updated existing entry for {enroll_no}"
    else:
        labels.append(enroll_no)
        embeddings_db = np.vstack([embeddings_db, mean_emb]) if embeddings_db.size else mean_emb[np.newaxis, :]
        msg = f"Registered new student {enroll_no}"

    # Save DB
    save_database(labels, embeddings_db, DB_PATH)
    return {"message": msg, "total_students": len(labels)}


# Recognition Terminal

DET_SIZE_RECO = 640
MATCH_THRESHOLD = 0.7

def load_database2(db_path):
    if not os.path.exists(db_path):
        return np.array([]), np.array([])
    data = np.load(db_path, allow_pickle=True)
    return data["labels"], data["embeddings"]

def l2_normalize2(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def match_face(embedding, embeddings_db, labels, threshold):
    if embeddings_db.size == 0:
        return "Unknown", 0.0
    sims = np.dot(embeddings_db, embedding)
    idx = np.argmax(sims)
    if sims[idx] >= threshold:
        return labels[idx], sims[idx]
    return "Unknown", sims[idx]

def annotate_faces(img, faces, labels, embeddings_db, threshold):
    recognized_ids = []
    unrec_img = img.copy()

    for face in faces:
        emb = l2_normalize2(face.embedding)
        name, score = match_face(emb, embeddings_db, labels, threshold)
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Annotate all faces
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{name} ({score:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # For unrecognized-only image
        if name == "Unknown":
            cv2.rectangle(unrec_img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(unrec_img, f"{name} ({score:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            recognized_ids.append(name)

    return recognized_ids, img, unrec_img

def img_to_base64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


def save_base64_image(base64_str, filename):
    """Save Base64 image string to file for debugging."""
    # Remove prefix if exists
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(filename, img)

@app.post("/recognize")
async def recognize_classroom(file: UploadFile = File(...)):
    labels, embeddings_db = load_database2(DB_PATH)
    app_insight = FaceAnalysis(name="buffalo_l")
    app_insight.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))

    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    faces = app_insight.get(img)
    rec_ids, all_faces_img, unrec_img = annotate_faces(img.copy(), faces, labels, embeddings_db, MATCH_THRESHOLD)

    result = {
        "recognized_ids": rec_ids,
        "annotated_all": img_to_base64(all_faces_img),
        "annotated_unrecognized": img_to_base64(unrec_img)
    }

    save_base64_image(result["annotated_all"], "debug_all_faces.jpg")
    save_base64_image(result["annotated_unrecognized"], "debug_unrec_faces.jpg")

    return JSONResponse(result)
