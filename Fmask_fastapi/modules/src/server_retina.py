import time
import uvicorn
from fastapi import FastAPI, File, UploadFile
from Fmask_fastapi.modules.src.predict_retina import retina_predict, retina_filter_mask, retina_NMS_apply, transform_image

app = FastAPI()

# Put an image and make a prediction
@app.post('/objectdetection/')
async def predict_api(file: UploadFile = File(...)):

    extension = file.filename.split('.')[-1] in ('jpg')

    if not extension:
        return 'Image must be jpg or png format.'

    content = await file.read()

    image = transform_image(content)
    start_time = time.time()
    prediction = retina_predict(image)
    print(f"Processing time: {time.time() - start_time}")
    print(prediction[0])

    threshold = 0.5
    prefinal_pr = prediction[0]
    prefinal_pr = retina_filter_mask(prefinal_pr, threshold)

    nms_threshold = 0.3
    final_pr = retina_NMS_apply(prefinal_pr, prediction[0], nms_threshold)

    boxes = final_pr['boxes'].tolist()
    labels = final_pr['labels'].tolist()
    scores = final_pr['scores'].tolist()

    response = {
        'boxes': boxes,
        'message': 'Prediction successful.',
        'labels': labels,
        'scores': scores,
    }
    print(boxes)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)