FROM python:3.9

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary files to the container
COPY retina_img.py Fmask_streamlit/retina_img.py

EXPOSE 8501
CMD ["streamlit", "run", "Fmask_streamlit/retina_img.py"]