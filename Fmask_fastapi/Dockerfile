FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary files to the container
COPY modules/src/predict_retina.py Fmask_fastapi/modules/src/predict_retina.py
COPY modules/src/server_retina.py Fmask_fastapi/modules/src/server_retina.py
COPY modules/RetinaNet_weights.pth Fmask_fastapi/modules/RetinaNet_weights.pth

# Expose the necessary port for FastAPI
EXPOSE 8080

CMD ["uvicorn", "Fmask_fastapi.modules.src.server_retina:app", "--host", "0.0.0.0", "--port", "8080"]