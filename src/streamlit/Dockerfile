FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -U pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt
# Copy the necessary files to the container
COPY stream_retina.py src/streamlit/stream_retina.py

EXPOSE 8501
CMD ["streamlit", "run", "src/streamlit/stream_retina.py"]