import torch
import torchvision

import numpy as np
#import albumentations as A
#from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms as T
from torchvision import datasets, models, transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from io import BytesIO
from PIL import Image

device = torch.device('cpu')

# convert bytes into an image
def transform_image(file):
    image = Image.open(BytesIO(file)).convert('RGB')
    return image

# load fasterrcnn model
def fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# get a model
def load_model():
    model = fasterrcnn_model(4).to(device)

    model = torch.load('/Users/burninggarment/mask_detection/Face_Mask_detection/fastapi/modules/model_weights.pth', map_location=device)
    model.eval()
    return model
    
def predict(image):
    transform = T.Compose([
        T.ToTensor()]
        )

    #image = np.array(image)
    image = transform(img=image)
    image  = torch.unsqueeze(image, 0)

    model = load_model()

    output = model(image)
    boxes = output[0]['boxes'].detach().cpu().numpy()
    labels = output[0]['labels'].detach().cpu().numpy()
    scores = output[0]['scores'].detach().cpu().numpy()

    return boxes, labels, scores
