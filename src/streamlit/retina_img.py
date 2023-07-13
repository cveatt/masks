import os
import streamlit as st
import requests
from PIL import Image, ImageDraw  
#mport boto3
#from botocore.exceptions import NoCredentialsError
import pandas as pd 
import numpy as np 
import json
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the retrieved values in your Streamlit application as needed
host = os.getenv('HOST')
port = os.getenv('PORT')
endpoint = os.getenv('ENDPOINT')
URL = f'{host}:{port}{endpoint}'

st.title('Object Detection')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(
    '<h1 style="text-align: center; color: #DDA0DD; ">Face Mask Recognition</h1>',
    unsafe_allow_html=True)

# AWS S3 bucket configuration
# input_bucket_name = 'face-mask-detection-input'
# output_bucket_name = 'face-mask-detection-output'

# class_colors = {
#     0: (0, 255, 255),
#     1: (0, 255, 0), 
#     2: (0, 0, 255),
#     3: (255, 0, 0)
# }

st.write('''

<p style="color:black;">This is a simple app for performing face mask detection. It utilizes a FastAPI service as the backend for object detection.</p>

<p style="color:grey;"><strong>Instructions:</strong></p>
<ol style="color:black;">
<li>Upload an image using the file uploader widget.</li>
<li>Click on the "Detect" button to run the face mask detection algorithm.</li>
<li>The image will be displayed with bounding boxes around detected faces indicating the presence of a mask.</li>
</ol>

<p style="color:black;"><a href="http://localhost:8180/docs" style="color:purple;">Click here to visit the FastAPI documentation</a> for more information on the backend service.</p>
''', unsafe_allow_html=True)


# Displays a file uploader widget
input_image = st.file_uploader('Load an image', type='jpg')

# def image_with_bbox(image, bboxes, blabels):
#     # Draw bounding boxes on image
#     image_with_box = Image.fromarray(image)
#     draw = ImageDraw.Draw(image_with_box)
    
#     #print(bboxes)
#     for bbox, label in zip(bboxes, blabels):
#         #print(bbox)
#         xmin, ymin, xmax, ymax = bbox
#         color = class_colors[label]
        
#         draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
#         draw.text((xmin, ymin), fill=color, text=class_colors)
#     return image_with_box

# def upload_to_s3(file_data, bucket_name, object_name):
#     s3 = boto3.client(
#         's3',
#         aws_access_key_id=os.getenv('AWS_KEY_ID'),
#         aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
#     )
#     try:
#         s3.put_object(Body=file_data, Bucket=bucket_name, Key=object_name)
#         return True
#     except NoCredentialsError:
#         st.error('AWS credentials not found. Make sure to configure AWS CLI or provide credentials through environment variables.')
#         return False
#     except Exception as e:
#         st.error(f'Error uploading file to S3: {str(e)}')
#         return False

            
# Displays a button
if st.button('Detect'):
    if input_image is not None:
        # image_bytes = input_image.tobytes()
        files = {'file': input_image}
        res = requests.post(URL, files=files)
        if res.status_code == 200:
            image = Image.open(input_image)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write('Just a second ...')
            logger.info("Response", res.content)
            st.image(res.content, caption='Image with Bounding Boxes', use_column_width=True)
        else:
            st.write(f"Error: {res.status_code} - {res.text}")

        # processed_image = Image.open(io.BytesIO(res.content))
        # predicted_class = json.loads(res.content.decode('utf-8'))
        # print('Predicted_classes:', predicted_class)

        # success = upload_to_s3(input_image.read(), input_bucket_name, 'image.jpg')
        # if not success:
        #     st.error('Failed to upload image to S3.')
        #     st.stop()

        # bboxes = predicted_class['boxes']
        # blabels = predicted_class['labels']

        # print(blabels)

        # image_with_boxes = image_with_bbox(np.array(image), bboxes, blabels)
        # image_with_boxes = Image.fromarray(np.uint8(image_with_boxes))
        #image_with_boxes = 
        # st.image(res.content, caption='Image with Bounding Boxes', use_column_width=True)