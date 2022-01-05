
import os
import numpy as np
import streamlit as st
from PIL import Image
import os
import cv2

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def save_uploaded_image(uploaded_image):
    
    try:
        with open(os.path.join('new',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False
    
    
# Model saved with Keras model.save()
MODEL_PATH ='model_vgg16.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   
    x = preprocess_input(x)

    preds = model.predict(x)
    preds=(preds*100)
    preds = preds.tolist()
    preds = [item for elem in preds for item in elem]
    preds = [round(num) for num in preds]
    if preds in range(50,100):
        pred="The Person is not Infected With Malaria"
    else:
        pred="The Person is Infected With Malaria"
    
    
    return pred
    
    
st.title('Malaria Detection')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)
        

if st.button('Predict'):
    display_image = Image.open(uploaded_image)
        
    # Make prediction
    pred = model_predict((os.path.join('new',uploaded_image.name)), model)
                              
    # display
    col1,col2 = st.columns(2)
        
    with col1:
        st.header('Your uploaded image')
        st.image(display_image)
    with col2:
        st.header(pred)          


        
