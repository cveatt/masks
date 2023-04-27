#import time
import streamlit as st
import requests
from PIL import Image
import json

url = 'http://localhost:8080'
endpoint = '/objectdetection'

st.title('Face Mask Detection')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(
    '<h1 style="text-align: center">Face_Mask_detection</h1>',
    unsafe_allow_html=True)

st.write('''
         
         This is a simple app for the Face Mask detection.
         
         This Streamlit example uses a FastAPI service as backend.
         
         Visit this URL at http://localhost:8080/docs for FastAPI documentation.
         
         ''')

classes = {
            0: '_',
            1: 'With Mask',
            2: 'Without Mask',
            3: 'Mask Weared Incorrect'
}
           

# Displays a file uploader widget
input_image = st.file_uploader('Load an image', type='jpg')

# Displays a button
if st.button('Get Predict'):
    if input_image is not None:
        files = {'file': input_image}
        res = requests.post(url + endpoint, files=files)
        image = Image.open(input_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write('Just a second ...')
        #time.sleep(3)
        predicted_class = json.loads(res.content.decode('utf-8'))
        #print(json.loads(res.content.decode('utf-8')))
        print(predicted_class)

        if predicted_class:
        #    predicted_class = int(predicted_class)
            st.write(f'Prediction: {predicted_class} !')
        #else:
         #   st.write('Error: Failed to get prediction')