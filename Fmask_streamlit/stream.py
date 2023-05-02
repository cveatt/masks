import streamlit as st
import requests
from PIL import Image, ImageDraw  
import pandas as pd 
import numpy as np 
import json

url = 'http://localhost:8080'
endpoint = '/objectdetection'

st.title('Face Mask Detection')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(
    '<h1 style="text-align: center">Face_Mask_detection</h1>',
    unsafe_allow_html=True)

classes = {
    0: '_',
    1: 'With Mask',
    2: 'Without Mask',
    3: 'Mask Worn Incorrect'
}

st.write('''
         
         This is a simple app for the Face Mask detection.
         
         This Streamlit example uses a FastAPI service as backend.
         
         Visit this URL at http://localhost:8080/docs for FastAPI documentation.
         
         ''')

# Displays a file uploader widget
input_image = st.file_uploader('Load an image', type='jpg')

def image_with_bbox(image, bboxes):
    # Draw bounding boxes on image
    image_with_box = Image.fromarray(image)
    draw = ImageDraw.Draw(image_with_box)
    for _, row in bboxes.iterrows():
        xmin, ymin, xmax, ymax = row[["xmin", "ymin", "xmax", "ymax"]]
                # label = row["labels"]
                # color = label_colors[label]
                # print(f"Drawing box with color {color} at coordinates ({xmin},{ymin})-({xmax},{ymax})")
        draw.rectangle([xmin, ymin, xmax, ymax], width=1)
    return image_with_box
            
# Displays a button
if st.button('Get Predict'):
    if input_image is not None:
        files = {'file': input_image}
        res = requests.post(url + endpoint, files=files)
        image = Image.open(input_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write('Just a second ...')
        predicted_class = json.loads(res.content.decode('utf-8'))
        #print(json.loads(res.content.decode('utf-8')))
        print("Predicted_classes:", predicted_class)

        xmin, xmax, ymin, ymax, labels = [], [], [], [], []

        # prediction = predicted_class['boxes']
        print(predicted_class)
        #print(prediction["boxes"])

        # for prediction_dict in predicted_class:
        for box in predicted_class['boxes']:
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            xmin.append(x)
            ymin.append(y)
            xmax.append(x + width)
            ymax.append(y + height)
            
        boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
        bboxes = boxes[["xmin", "ymin", "xmax", "ymax"]]
        label_colors = {
                        'With Mask': (0, 255, 0),
                        'Without Mask': (255, 0, 0), 
                        "Mask Worn Incorrect": (0, 0, 255)
                    }

    image_with_boxes = image_with_bbox(np.array(image), bboxes)
    image_with_boxes = Image.fromarray(np.uint8(image_with_boxes))
    st.image(image_with_boxes, caption="Image with Bounding Boxes", use_column_width=True)