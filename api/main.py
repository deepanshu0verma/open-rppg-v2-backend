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
model = rppg.Model()

def get_rr(y, sr=30):
    """
    Extracts Respiratory Rate from the BVP signal using Welch's PSD.
    Focuses on the 0.1Hz to 0.5Hz range.
    """
    # Convert list to numpy array so SciPy can process it
    y_arr = np.array(y)
    
    if len(y_arr) < sr * 3:
        return 0.0
    
    # Use y_arr instead of y
    p, q = welch(y_arr, sr, nfft=20000, nperseg=len(y_arr))
    
    rr_band = (p >= 0.1) & (p <= 0.5)
    
    if not any(rr_band):
        return 0.0
        
    peak_freq = p[rr_band][np.argmax(q[rr_band])]
    return float(peak_freq * 60)

@app.get("/api/health")
async def health_check():
    """
    The frontend will call this repeatedly until it gets a 200 OK.
    By the time this is reachable, the 'model = rppg.Model()' 
    initialization is finished.
    """
    return {"status": "ready"}

@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...)):
    start_time = time.time()
    
    # Save the chunk sent by React
    temp_path = f"temp_{int(time.time())}.webm"
    with open(temp_path, "wb") as f:
        f.write(await video.read())

    # Process video with OpenCV
    cap = cv2.VideoCapture(temp_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        img = cv2.resize(frame, (128, 128))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Clean up file immediately
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if len(frames) == 0:
        return {"bpm": 0, "rr": 0, "latency": "0.0s", "quality": 0}

    # 1. Run model inference ONCE
    video_tensor = np.array(frames)
    result = model.process_video_tensor(video_tensor) 
    
 # 2. Extract BVP for Respiratory Rate calculation
    bvp_signal, _ = model.bvp(raw=True)
    
    calculated_rr = 0.0
    if bvp_signal and len(bvp_signal) > 0:
        # get_rr now handles the conversion to numpy array internally
        calculated_rr = get_rr(bvp_signal, sr=model.fps)

    # Calculate total latency for this request
    total_latency = time.time() - start_time

    face_box = None
    if model.box is not None:
    # Use the filtered box for smoothness
     face_box = {
        "ymin": int(model.box[0][0]),
        "ymin_max": int(model.box[0][1]),
        "xmin": int(model.box[1][0]),
        "xmin_max": int(model.box[1][1])
    }

    return {
        "bpm": round(float(result.get('hr', 0)), 2),
        "rr": round(float(calculated_rr), 2),
        "latency": f"{total_latency:.2f}s",
        "quality": round(float(result.get('SQI', 0)), 2),
        "box": face_box
    }