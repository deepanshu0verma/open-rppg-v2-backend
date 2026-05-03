from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import time
import sys
from scipy.signal import welch
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rppg

app = FastAPI()

# Buffer to store ONLY high-quality readings for our aggregate
bpm_history = deque(maxlen=12)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


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
    bbox = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if bbox is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

            if len(faces) > 0:
                x, y, w_face, h_face = faces[0]
                # Keep the 15% margin to ensure full skin capture
                margin_y, margin_x = int(h_face * 0.15), int(w_face * 0.15)
                bbox = (
                    max(0, x - margin_x),
                    max(0, y - margin_y),
                    w_face + (margin_x * 2),
                    h_face + (margin_y * 2),
                )
            else:
                continue

        x, y, w_f, h_f = bbox
        h, w, _ = frame.shape
        roi = frame[y : min(y + h_f, h), x : min(x + w_f, w)]

        if roi.size != 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
            img = cv2.resize(roi, (128, 128))
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if len(frames) < 30:
        return {
            "chunk_bpm": 0,
            "aggregate_bpm": np.median(bpm_history) if bpm_history else 0,
            "rr": 0,
            "latency": f"{time.time() - start_time:.2f}s",
            "quality": 0,
        }

    video_tensor = np.array(frames)
    result = model.process_video_tensor(video_tensor)

    hr_raw, sqi_raw = result.get("hr"), result.get("SQI")

    raw_bpm = float(hr_raw) if hr_raw is not None else 0.0
    quality = float(sqi_raw) if sqi_raw is not None else 0.0

    # ==========================================
    # ENTERPRISE STABILITY ALGORITHM
    # ==========================================
    reported_chunk_bpm = raw_bpm

    # 1. Quality Gate: If signal is garbage (< 20%), inherit last known good BPM
    if quality < 0.20 and bpm_history:
        reported_chunk_bpm = np.median(bpm_history)

    # 2. Harmonic Rejection: Prevent the 41 BPM halving drop or 110 BPM spikes
    if bpm_history:
        current_median = np.median(bpm_history)
        # If reading is weirdly low (halving) or weirdly high (spiking), reject it
        if reported_chunk_bpm < (current_median * 0.65) or reported_chunk_bpm > (
            current_median * 1.35
        ):
            reported_chunk_bpm = current_median

    # 3. Add to History ONLY if the final logic deems it valid
    if 40 < reported_chunk_bpm < 150:
        bpm_history.append(reported_chunk_bpm)

    running_aggregate = np.median(bpm_history) if bpm_history else reported_chunk_bpm

    # Biomarker: Respiratory Rate
    bvp_signal, _ = model.bvp(raw=True)
    calculated_rr = (
        get_rr(bvp_signal, sr=model.fps) if bvp_signal and len(bvp_signal) > 0 else 0.0
    )

    return {
        "chunk_bpm": round(reported_chunk_bpm, 2),
        "aggregate_bpm": round(running_aggregate, 2),
        "rr": round(float(calculated_rr), 2),
        "latency": f"{time.time() - start_time:.2f}s",
        "quality": round(quality * 100, 1),
    }
