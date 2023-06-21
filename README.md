# Face Mask Recognition
## Description
This is a face mask detection project made in Kaggle by using [PyTorch](https://pytorch.org) with [RetinaNet](https://pytorch.org/vision/main/models/retinanet.html) pretrained model for object detection. It helps to recognize whether there is a person in the picture wearing a medical mask or not. 
## Instruction
+ Use Kaggle [face mask dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) with images and annotations.
+ Put model weights into (/masks/Fmask_fastapi/modules/src) folder. These weights you can get by using my training [notebook](https://www.kaggle.com/code/cveatt/facemask-recognition-w-kaggle-data).
+ Use [RETINANET_RESNET50_FPN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection).
+ (OPTIONAL) You can also use a custom datset of [office folks](https://www.kaggle.com/datasets/cveatt/office-masks) and with weights of pretrained on the Kaggle data modify the model.

# Run FastAPI
```
uvicorn src/fastapi/server_retina:app --host=0.0.0.0 --port=8888
```
# Run Streamlit
## Application for image detection
```
streamlit run src/streamlit/retina_img.py
```
## Application for video detection
```
streamlit run src/streamlit/stream_retina.py
```
# Docker
## Build the Docker image for FastAPI 
```
cd src/fastapi/
docker build -t fastapi -f Dockerfile .
```
## Run the Docker container for FastAPI
```
docker run -p 8888:8888 fastapi
```
## Build the Docker image for Streamlit
```
cd src/streamlit/
docker build -t streamlit -f Dockerfile .
```
## Run the Docker container for Streamlit
```
docker run -p 8585:8501 streamlit
```
# Docker Compose
## Building
```
docker-compose build
``` 
## Run Docker Compose
```
docker-compose up
```
## Link to the Kaggle notebook:
https://www.kaggle.com/code/cveatt/facemask-recognition-w-kaggle-data
https://www.kaggle.com/code/cveatt/facemask-recognition-w-custom-data/notebook
