import streamlit as st
import cv2
import numpy as np
import requests
import json
from streamlit_webrtc import webrtc_streamer
import av
import time


url = 'http://localhost:8080'
endpoint = '/objectdetection'

# RGB
class_colors = {
    0: (0, 255, 255), # yellow
    1: (0, 255, 0), # green
    2: (0, 0, 255), # blue
    3: (255, 0, 0) # red
}

def image_with_bbox(image, bboxes, blabels):
    # Draw bounding boxes on image
    image_with_box = image.copy()
    for bbox, label in zip(bboxes, blabels):
        xmin, ymin, xmax, ymax = map(int, bbox)
        color = tuple(map(int, class_colors[label]))
        
        cv2.rectangle(image_with_box, (xmin, ymin), (xmax, ymax), color, thickness=2)
    return image_with_box

# Function to process video frames
def record_video(video_frame):
    # Convert frame to OpenCV format
    img = video_frame.to_ndarray(format='bgr24')
    new_size = (480, 400)
    img = cv2.resize(img, new_size)

    # Send frame to object detection API for inference
    start_time = time.time()
    _, img_encoded = cv2.imencode('.jpg', img)
    response = requests.post(url + endpoint, files={'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')})
    predictions = json.loads(response.content.decode('utf-8'))
    print(f"Processing time: {time.time() - start_time}")

    bboxes = predictions['boxes']
    blabels = predictions['labels']

    print("Here are labels", blabels)

    # Overlay bounding boxes and labels on frame and display in Streamlit app
    frame_with_boxes = image_with_bbox(img, bboxes, blabels)
    #st.image(frame_with_boxes, channels='BGR')
    #rgb_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
    return av.VideoFrame.from_ndarray(frame_with_boxes, format='bgr24')
    #return frame_with_boxes

# Set up Streamlit app with WebRTC streamer
def main():
    st.title('Real-Time Object Detection')
    webrtc_streamer(key="example", video_frame_callback=record_video)

if __name__ == '__main__':
    main()