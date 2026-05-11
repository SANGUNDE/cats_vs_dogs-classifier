import streamlit as st
import numpy as np
import tensorflow as tf  
import keras 
from PIL import Image

st.title('Cats and Dogs Classifier')

@st.cache_resource
def load_my_model():
    return keras.models.load_model("cats_vs_dogs.keras")

model = load_my_model()

uploaded_file = st.file_uploader(label='UPLOAD AN IMAGE', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

   
    def process_img(img):
        img = img.convert('RGB') 
        img = img.resize((224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 
        return img_array

    processed_image = process_img(image)

    
    prediction = model.predict(processed_image)
    
    label = 'Dog' if prediction > 0.5 else 'Cat'

    st.header(f'Predicted Result: {label}')