import torch
import torchvision

import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from io import BytesIO
from PIL import Image

device = torch.device('cpu')

# Convert bytes into an image
def transform_image(file):
    image = Image.open(BytesIO(file)).convert('RGB')
    image = np.array(image)
    return image

# Load fasterrcnn model
def fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

num_classes = 4
# Get a model
def load_model():
    model = fasterrcnn_model(num_classes).to(device)

    model = torch.load('Fmask_fastapi/modules/Custom_model_weights.pth', map_location=device)
    model.eval()
    return model

model = load_model()
def predict(image):
    transform = A.Compose([ToTensorV2()])
    image = transform(image=image)['image']
    image  = torch.unsqueeze(image, 0)
    image = image.float() / 255.0
    with torch.no_grad():
        output = model(image)
    return output

def filter_mask(prefinal_pr, threshold):
    my_filter = prefinal_pr['scores'] > threshold
    prefinal_pr['boxes'] = prefinal_pr['boxes'][my_filter]
    prefinal_pr['scores'] = prefinal_pr['scores'][my_filter]
    prefinal_pr['labels'] = prefinal_pr['labels'][my_filter]
    return prefinal_pr

def NMS_apply(prefinal_pr, threshold):
    keep = torchvision.ops.nms(prefinal_pr['boxes'], prefinal_pr['scores'], threshold)

    prefinal_pr['boxes'] = prefinal_pr['boxes'][keep]
    prefinal_pr['scores'] = prefinal_pr['scores'][keep]
    prefinal_pr['labels'] = prefinal_pr['labels'][keep]
    return prefinal_pr