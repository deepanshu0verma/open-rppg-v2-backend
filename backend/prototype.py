import rppg
import cv2
import numpy as np
import time
import os

def run_prototype(video_path):
    # 1. Initialize the Open-rPPG Model
    # This automatically loads FacePhys or the default model
    try:
        model = rppg.Model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0 # Default fallback
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_5s = int(fps * 5)
    
    chunk_results = []
    frame_buffer = []
    
    print(f"--- Prototype Started ---")
    print(f"Video: {video_path} | FPS: {fps:.2f} | Total Frames: {total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Collect frames for the 5-second chunk
        frame_buffer.append(frame)

        if len(frame_buffer) == frames_per_5s:
            start_time = time.time()
            
            # 2. Near Real-Time Processing (Incremental)
            # Convert buffer to a numpy tensor [Time, Height, Width, Channels]
            video_tensor = np.array(frame_buffer)
            
            # Using the toolbox's built-in tensor processor
            # This handles face cropping + HR + Respiratory Rate
            result = model.process_video_tensor(video_tensor, fps=fps)
            
            latency = time.time() - start_time
            
            # Extract metrics (BPM and RR)
            bpm = result.get('hr', 0)
            rr = result.get('breathingrate', 0) # Bonus: Respiratory Rate
            sqi = result.get('SQI', 0)          # Signal Quality Index
            
            chunk_results.append(bpm)
            
            print(f"Chunk {len(chunk_results)} | BPM: {bpm:.1f} | RR: {rr:.1f} | Latency: {latency:.2f}s | Quality: {sqi:.2f}")
            
            # Clear buffer for next 5s chunk
            frame_buffer = []

    cap.release()

    # 3. Final Aggregation
    if chunk_results:
        final_bpm = np.mean(chunk_results)
        print(f"\n--- FINAL ESTIMATION ---")
        print(f"Full 60s BPM Estimate: {final_bpm:.2f}")
    else:
        print("Error: No data processed. Check if the video contains a visible face.")

if __name__ == "__main__":
    # Ensure you have a video file in this folder
    test_file = "test_video.mp4" 
    if os.path.exists(test_file):
        run_prototype(test_file)
    else:
        print(f"File {test_file} not found. Please add a 60s video to the folder.")