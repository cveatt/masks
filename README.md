# Face Mask Recognition
## Description
This is a face mask detection project made in Kaggle by using [PyTorch](https://pytorch.org) with [FasterRCNN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) or [RetinaNet](https://pytorch.org/vision/main/models/retinanet.html) pretrained model for object detection. It helps to recognize whether there is a person in the picture wearing a medical mask or not. 
## Instruction
+ Use Kaggle [face mask dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) with images and annotations.
+ Put model weights into (/masks/Fmask_fastapi/modules/src) folder. These weights you can get by using my training [notebook](https://www.kaggle.com/code/cveatt/facemask-recognition-w-kaggle-data)
+ Use [FASTERRCNN_RESNET50_FPN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) or [RETINANET_RESNET50_FPN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection).
+ (OPTIONAL) You can also use a custom datset of [office folks](https://www.kaggle.com/datasets/cveatt/office-masks) and with weights of pretrained on the Kaggle data modify the model.
## Run FastAPI for FasterRCNN model
```
uvicorn Fmask_fastapi.modules.src.server:app --host=0.0.0.0 --port=8080
```
## Run FastAPI for RetinaNet model
```
uvicorn Fmask_fastapi.modules.src.server_retina:app --host=0.0.0.0 --port=8080
```
## Run Streamlit for FasterRCNN model
```
streamlit run Fmask_streamlit/stream.py
```
```
streamlit run Fmask_streamlit/stream_in_rl.py
```
## Run Streamlit for RetinaNet model
```
streamlit run Fmask_streamlit/retina_img.py
```
```
streamlit run Fmask_streamlit/stream_retina.py
```
# Docker
## Build the Docker image for FastAPI 
```
cd Fmask_fastapi/
docker build -t fastapi -f Dockerfile .
```
## Run the Docker container for FastAPI
```
docker run -p 8080:8080 fastapi
```
## Build the Docker image for Streamlit
```
cd Fmask_streamlit/
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
