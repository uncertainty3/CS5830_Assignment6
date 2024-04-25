from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model as keras_load_model
import sys, uvicorn

IMAGEDIR = "images/"
app = FastAPI()

model_path = sys.argv[1]
#model_path = "C:\\Users\\91990\Downloads\\FastAPI\\mnist-epoch-10.hdf5"

def load_model(model_path):
    model = keras_load_model(model_path)
    return model

def format_image(image):
    # Open the image using PIL
    # Convert the image to grayscale
    image = image.convert("L")
    #image = ImageOps.invert(image)
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Flatten the image array to a 1D array of 784 elements
    flattened_image = image_array.reshape(1, 784)
    # Normalize the pixel values to the range [0, 1]
    normalized_image = flattened_image / 255.0
    return normalized_image

def predict(img):
    #print(x.shape)
    x = format_image(img)
    model = load_model(model_path)
    prediction = model.predict(x)
    digit = np.argmax(prediction)
    return str(digit)


@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
 
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
 
    #save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
    
    img = Image.open(f'{IMAGEDIR}{file.filename}')

    ans = predict(img)

    os.remove(f'{IMAGEDIR}{file.filename}')

    return {"digit": ans}

if __name__ == "__main__":
    uvicorn.run(app)