FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -U pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary files to the container
COPY predict_retina.py src/fastapi/predict_retina.py
COPY server_retina.py src/fastapi/server_retina.py
COPY post_processing.py src/fastapi/post_processing.py
COPY weights/RetinaNet_weights.pth src/fastapi/weights/RetinaNet_weights.pth

# Expose the necessary port for FastAPI
EXPOSE 8180

CMD ["uvicorn", "src.fastapi.server_retina:app", "--host", "0.0.0.0", "--port", "8180"]