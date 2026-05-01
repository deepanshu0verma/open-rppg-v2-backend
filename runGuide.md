1. Activate your Virtual Environment: 
source venv/bin/activate

2. If you don't have a venv yet, create it now
If the command above fails, create a fresh one:
python3.11 -m venv venv
source venv/bin/activate

3. Install your requirements
Now pip will be able to see the file:
pip install --upgrade pip
pip install -r requirements.txt

4. Run with the Python Path fixed
Uvicorn needs to know that the current directory is where api lives. Run it like this:
export PYTHONPATH=$PYTHONPATH:.

5. Install Required Libraries :
python3 -m pip install fastapi uvicorn python-multipart torch torchvision numpy opencv-python-headless jax jaxlib keras einops av heartpy onnxruntime scipy

6. Launch the Backend Server :
python3 -m uvicorn api.main:app --reload --port 8000 --host 0.0.0.0

7. Keep this terminal running. You should see "Uvicorn running on http://0.0.0.0:8000".