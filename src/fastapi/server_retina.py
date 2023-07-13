import os
import time
import io
import datetime
import boto3
from botocore.exceptions import NoCredentialsError
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, Response
from src.fastapi.predict_retina import retina_predict, retina_filter_mask, retina_NMS_apply, transform_image
from src.fastapi.post_processing import image_with_bbox

# AWS S3 bucket configuration
input_bucket_name = 'face-mask-detection-input'
output_bucket_name = 'face-mask-detection-output'

app = FastAPI()

app.s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
    )
# Put an image and make a prediction
@app.post('/objectdetection/')
async def predict_api(file: UploadFile = File(...)):
    content = await file.read()

    image = transform_image(content)
    start_time = time.time()
    prediction = retina_predict(image)
    print(f'Processing time: {time.time() - start_time}')

    threshold = 0.5
    prefinal_pr = prediction[0]
    prefinal_pr = retina_filter_mask(prefinal_pr, threshold)

    nms_threshold = 0.3
    final_pr = retina_NMS_apply(prefinal_pr, nms_threshold)

    boxes = final_pr['boxes'].tolist()
    labels = final_pr['labels'].tolist()
    scores = final_pr['scores'].tolist()

    processed_img = image_with_bbox(image, boxes, labels)

    output_buffer = io.BytesIO()
    processed_img.save(output_buffer, format='JPEG')
    processed_img_output = output_buffer.getvalue()

    current_time = datetime.datetime.now().astimezone().isoformat()
    file_name = f'image_{current_time}.jpeg'

    upload_input = upload_to_s3(image.tobytes(), input_bucket_name, file_name)
    processed_img_to_s3 = processed_img.copy().tobytes()
    
    upload_output = upload_to_s3(processed_img_to_s3, output_bucket_name, file_name)

    return StreamingResponse(io.BytesIO(processed_img_output), media_type='image/jpg')

def upload_to_s3(file_data, bucket_name, object_name):
    try:
        app.s3.put_object(Body=file_data, Bucket=bucket_name, Key=object_name)
        return True
    except NoCredentialsError:
        print('AWS credentials not found.')
        return False
    except Exception as e:
        print(f'Error uploading file to S3: {str(e)}')
        return False