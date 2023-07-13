from PIL import Image, ImageDraw
#import numpy as np

class_colors = {
    0: (0, 255, 0),
    1: (255, 0, 0), 
    2: (255, 255, 0),
}

def image_with_bbox(image, bboxes, blabels):
    # Draw bounding boxes on image
    image_with_box = Image.fromarray(image)
    draw = ImageDraw.Draw(image_with_box)
    
    for bbox, label in zip(bboxes, blabels):
        xmin, ymin, xmax, ymax = bbox
        color = class_colors[label]
        
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        draw.text((xmin, ymin), fill=color, text=class_colors)
    return image_with_box