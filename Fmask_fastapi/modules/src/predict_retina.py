import torch
import torchvision

import math
import numpy as np
import albumentations as A
from torch import nn
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

from io import BytesIO
from PIL import Image

device = torch.device('cpu')

# Convert bytes into an image
def transform_image(file):
    image = Image.open(BytesIO(file)).convert('RGB')
    image = np.array(image)
    return image

# Load the pretrained RetinaNet model
def retinanet_model(num_classes):
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=torchvision.models.detection.retinanet.RetinaNet_ResNet50_FPN_Weights.COCO_V1,
    )
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head.num_classes = num_classes

    cls_logits = nn.Conv2d(256, num_anchors * num_classes, kernel_size = 3, stride = 1, padding = 1)
    torch.nn.init.normal_(cls_logits.weight, std = 0.01)  # RetinaNetClassificationHead
    torch.nn.init.constant_(cls_logits.bias, - math.log((1 - 0.01) / 0.01))  # RetinaNetClassificationHead
    
    model.head.classification_head.cls_logits = cls_logits
    return model

num_classes = 3
# Get a model
def load_retina_model():
    model = retinanet_model(num_classes).to(device)

    model = torch.load('Fmask_fastapi/modules/RetinaNet_weights.pth', map_location=device)
    model.eval()
    return model

model = load_retina_model()

def retina_predict(image):
    transform = A.Compose([ToTensorV2()])
    image = transform(image=image)['image']
    image  = torch.unsqueeze(image, 0)
    image = image.float() / 255.0
    with torch.no_grad():
        output = model(image)
    return output

def retina_filter_mask(prefinal_pr, threshold):
    my_filter = prefinal_pr['scores'] > threshold
    prefinal_pr['boxes'] = prefinal_pr['boxes'][my_filter]
    prefinal_pr['scores'] = prefinal_pr['scores'][my_filter]
    prefinal_pr['labels'] = prefinal_pr['labels'][my_filter]
    return prefinal_pr

def retina_NMS_apply(prefinal_pr, threshold):
    keep = torchvision.ops.nms(prefinal_pr['boxes'], prefinal_pr['scores'], threshold)

    prefinal_pr['boxes'] = prefinal_pr['boxes'][keep]
    prefinal_pr['scores'] = prefinal_pr['scores'][keep]
    prefinal_pr['labels'] = prefinal_pr['labels'][keep]
    return prefinal_pr