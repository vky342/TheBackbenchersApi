
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2, numpy as np, base64, os
from insightface.app import FaceAnalysis

DB_PATH = "students_db.npz"
DET_SIZE = 640
MATCH_THRESHOLD = 0.7

app = FastAPI()

def load_database(db_path):
    if not os.path.exists(db_path):
        return np.array([]), np.array([])
    data = np.load(db_path, allow_pickle=True)
    return data["labels"], data["embeddings"]

def l2_normalize(x, axis=-1, eps=1e-8):
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
        emb = l2_normalize(face.embedding)
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
    labels, embeddings_db = load_database(DB_PATH)
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
