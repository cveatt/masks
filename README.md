# Face Mask Recognition
## Description
This is a face mask detection project made in Kaggle by using https://pytorch.org with https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html or (https://pytorch.org/vision/main/models/retinanet.html) pretrained model for object detection. It helps to recognize whether there is a person in the picture wearing a medical mask or not. 
## Instruction
+ Use Kaggle face mask dataset with images and annotations (https://www.kaggle.com/datasets/andrewmvd/face-mask-detection).
+ Use FASTERRCNN_RESNET50_FPN: (https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) or RETINANET_RESNET50_FPN: (https://pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn.html#torchvision.models.detection.retinanet_resnet50_fpn) to train face mask detection project.
+ (OPTIONAL) You can also use a custom datset of office folks (https://www.kaggle.com/datasets/cveatt/office-masks) and with weights of pretrained on the Kaggle data modify the model.
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
streamlit run Fmask_streamlit/stream_retina.py
```
## Link to the Kaggle notebook:
https://www.kaggle.com/code/cveatt/facemask-recognition-w-kaggle-data
https://www.kaggle.com/code/cveatt/facemask-recognition-w-custom-data/notebook
