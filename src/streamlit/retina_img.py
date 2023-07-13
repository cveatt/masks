import os
import streamlit as st
import requests
from PIL import Image, ImageDraw  
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
            
# Displays a button
if st.button('Detect'):
    if input_image is not None:
        files = {'file': input_image}
        res = requests.post(URL, files=files)
        if res.status_code == 200:
            image = Image.open(input_image)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write('Just a second ...')
            st.image(res.content, caption='Image with Bounding Boxes', use_column_width=True)
        else:
            st.write(f'Error: {res.status_code} - {res.text}')