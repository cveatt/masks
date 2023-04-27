import uvicorn
from fastapi import FastAPI, File, UploadFile
from predict import predict, transform_image

app = FastAPI()

# put an image and make a prediction
@app.post('/objectdetection/')
async def predict_api(file: UploadFile = File(...)):

    extension = file.filename.split('.')[-1] in ('jpg')

    if not extension:
        return 'Image must be jpg or png format.'

    content = await file.read()

    image = transform_image(content)
    prediction = predict(image)
    print(prediction[0])
    labels = prediction[0]['labels'].tolist()
    pred_value = prediction[0]['boxes'].tolist()
    
    response = {
        #'prediction': prediction,
        'boxes': pred_value,
        'message': 'Prediction successful.',
        'labels': labels,
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)
    