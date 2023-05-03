import streamlit as st
import requests
from PIL import Image, ImageDraw  
import pandas as pd 
import numpy as np 
import json

url = 'http://localhost:8080'
endpoint = '/objectdetection'


st.title('Object Detection')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(
    '<h1 style="text-align: center; color: #DDA0DD; ">Face Mask Recognition</h1>',
    unsafe_allow_html=True)

class_colors = {
    0: (255, 255, 0),
    1: (0, 255, 0), 
    2: (255, 0, 0),
    3: (0, 0, 255)
}

st.write('''

<p style="color:black;">This is a simple app for performing face mask detection. It utilizes a FastAPI service as the backend for object detection.</p>

<p style="color:grey;"><strong>Instructions:</strong></p>
<ol style="color:black;">
<li>Upload an image using the file uploader widget.</li>
<li>Click on the "Detect" button to run the face mask detection algorithm.</li>
<li>The image will be displayed with bounding boxes around detected faces indicating the presence of a mask.</li>
</ol>

<p style="color:black;"><a href="http://localhost:8080/docs" style="color:purple;">Click here to visit the FastAPI documentation</a> for more information on the backend service.</p>
''', unsafe_allow_html=True)


# Displays a file uploader widget
input_image = st.file_uploader('Load an image', type='jpg')

def image_with_bbox(image, bboxes, blabels):
    # Draw bounding boxes on image
    image_with_box = Image.fromarray(image)
    draw = ImageDraw.Draw(image_with_box)
    
    print(bboxes)
    for bbox, label in zip(bboxes, blabels):
        print(bbox)
        xmin, ymin, xmax, ymax = bbox
        color = class_colors[label]
        
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=1)
        draw.text((xmin, ymin), fill=color, text=class_colors)
    return image_with_box
            
# Displays a button
if st.button('Detect'):
    if input_image is not None:
        files = {'file': input_image}
        res = requests.post(url + endpoint, files=files)
        image = Image.open(input_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write('Just a second ...')
        predicted_class = json.loads(res.content.decode('utf-8'))
        print("Predicted_classes:", predicted_class)

        bboxes = predicted_class['boxes']
        blabels = predicted_class['labels']

        image_with_boxes = image_with_bbox(np.array(image), bboxes, blabels)
        image_with_boxes = Image.fromarray(np.uint8(image_with_boxes))
        st.image(image_with_boxes, caption="Image with Bounding Boxes", use_column_width=True)