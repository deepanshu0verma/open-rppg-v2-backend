1. Install Required Libraries :
pip install fastapi uvicorn python-multipart torch torchvision numpy opencv-python-headless jax jaxlib keras einops av heartpy onnxruntime scipy

2. Launch the Backend Server :
python -m uvicorn api.main:app --reload --port 8000 --host 0.0.0.0

3. Keep this terminal running. You should see "Uvicorn running on http://0.0.0.0:8000".