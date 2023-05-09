import streamlit as st
import cv2
import numpy as np
import requests
import json

url = 'http://localhost:8080'
endpoint = '/objectdetection'

st.title('Real-Time Object Detection')

# Initialize camera capture object
cap = cv2.VideoCapture(0) # 0 - image from local web camera
stop_button = st.button('Stop')

# Dictionary in BGR
class_colors = {
    0: (0, 255, 255),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 0, 0)
}

def image_with_bbox(image, bboxes, blabels):
    # Draw bounding boxes on image
    image_with_box = image.copy()
    for bbox, label in zip(bboxes, blabels):
        xmin, ymin, xmax, ymax = map(int, bbox)
        color = tuple(map(int, class_colors[label]))
        
        cv2.rectangle(image_with_box, (xmin, ymin), (xmax, ymax), color, thickness=2)
        #cv2.putText(image_with_box, class_colors[label], (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
    return image_with_box

# Loop to capture frames and perform object detection on each frame
while cap.isOpened() and not stop_button:
    ret, frame = cap.read()  # ret - Boolean 

    if ret:
        # Send frame to object detection API for inference
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(url + endpoint, files={'file': ('image.jpg', img_encoded.tostring(), 'image/jpeg')})
        predictions = json.loads(response.content.decode('utf-8'))

        bboxes = predictions['boxes']
        blabels = predictions['labels']

        # Overlay bounding boxes and labels on frame and display in Streamlit app
        frame_with_boxes = image_with_bbox(frame, bboxes, blabels)
        st.image(frame_with_boxes, channels='BGR')

    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
        break

# Release camera capture object and close Streamlit app
cap.release()
cv2.destroyAllWindows()
