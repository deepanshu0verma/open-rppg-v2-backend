from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import time
import sys
from scipy.signal import welch

# Ensure Python can find your local rppg folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rppg 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the rPPG model
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = rppg.Model()

def get_rr(y, sr=30):
    y_arr = np.array(y)
    if len(y_arr) < sr * 3:
        return 0.0
    p, q = welch(y_arr, sr, nfft=20000, nperseg=len(y_arr))
    rr_band = (p >= 0.1) & (p <= 0.5)
    if not any(rr_band):
        return 0.0
    peak_freq = p[rr_band][np.argmax(q[rr_band])]
    return float(peak_freq * 60)

@app.get("/api/health")
async def health_check():
    return {"status": "ready"}

@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...)):
    start_time = time.time()
    
    temp_path = f"temp_{int(time.time())}.webm"
    with open(temp_path, "wb") as f:
        f.write(await video.read())

    cap = cv2.VideoCapture(temp_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None: 
            break
        h, w, _ = frame.shape
        forehead = frame[int(h*0.05):int(h*0.6), int(w*0.05):int(w*0.95)]
        
        if forehead.size != 0:
            img = cv2.resize(forehead, (128, 128))
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
    cap.release()
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if len(frames) < 10:
        return {"bpm": 0, "rr": 0, "latency": "0.0s", "quality": 0}

    # 1. Run model inference
    video_tensor = np.array(frames)
    result = model.process_video_tensor(video_tensor) 
    
    # --- SAFETY: Handle None values from the model ---
    hr_raw = result.get('hr')
    sqi_raw = result.get('SQI')
    
    # Convert safely; if None, use 0
    bpm = float(hr_raw) if hr_raw is not None else 0.0
    quality = float(sqi_raw) if sqi_raw is not None else 0.0

    # 2. Extract BVP for Respiratory Rate
    bvp_signal, _ = model.bvp(raw=True)
    calculated_rr = 0.0
    if bvp_signal and len(bvp_signal) > 0:
        calculated_rr = get_rr(bvp_signal, sr=model.fps)

    total_latency = time.time() - start_time

    face_box = None
    if model.box is not None:
        face_box = {
            "ymin": int(model.box[0][0]),
            "ymin_max": int(model.box[0][1]),
            "xmin": int(model.box[1][0]),
            "xmin_max": int(model.box[1][1])
        }

    return {
        "bpm": round(bpm, 2),
        "rr": round(float(calculated_rr), 2),
        "latency": f"{total_latency:.2f}s",
        "quality": round(quality, 2),
        "box": face_box
    }