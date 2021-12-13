############# DEPENDENCIES ##############

import streamlit as st

import numpy as np

# Importing and processing an image from the user.
from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.image import resize_with_pad

# Lodading the saved model.
from tensorflow.keras.models import load_model

############# STREAMLIT APP ##############

# This is my streamlit app for using a trained neural net to identify falling snow in images.

# The following links were useful for learning how to upload and work with an image in streamlit:
# - https://blog.jcharistech.com/2020/11/08/working-with-file-uploads-in-streamlit-python/
# - https://github.com/keras-team/keras/issues/11684
# - https://stackoverflow.com/questions/58900468/what-is-the-4th-channel-in-an-image

def read_file():
    # Returns image uploaded by user.
    return st.file_uploader("Please upload an image:", type=['bmp', 'jpg', 'jpeg', 'png', 'gif', 'tiff'])

def predict_and_show(image_bytes, snow_cutoff):
    
    # Convert bytes data to PIL image.
    image_raw = Image.open(image_bytes)
    
    # Convert image to RGB if it is not already, since the model was trained on RGB images.
    image = image_raw.convert('RGB')
    
    # Convert PIL image to an array Tensorflow can work with.
    image_array = img_to_array(image)
    
    # Replicate preprocessing done to images model was trained on.
    image_array = resize_with_pad(image_array, 640, 640)
    image_array = (image_array - 127.5) / 127.5

    # Add a dummy axis where the training images had an axis for batch.
    image_array = image_array[np.newaxis, :]

    # Load the trained model.
    model = load_model(f'../saved_models/test1')

    # Use trained model to predict presence or absence of snow.
    prediction = model.predict(image_array)[0][0]

    st.write(prediction) # **** REMOVE THIS
    
    # Display the prediciton.
    if prediction < snow_cutoff:
        st.write('The neural net has decided it is: not snowing.')
    else:
        st.write('The neural net has decided it is: snowing.')

    # Display the uploaded image under the prediction.
    st.image(image_raw)
    
def main():
    
    st.title('Neural Net for Snow Detection')

    st.write('This is my Capstone Project for the General Assembly Data Science Incubator.')
    st.write('Upload an image, and the neural net will tell you whether or not it appears to be snowing.')
    st.write('This and my other data science projects can be found on my GitHub repo [here](https://github.com/MarkCHarris).')
    st.write('Also see my [portfolio](https://markcharris.github.io).')
    st.write('The data used to train this neural net is from [here](https://sites.google.com/view/yunfuliu/desnownet).')
    
    snow_cutoff = 0.5
    
    # Prompt user to upload an image and return it.
    image_bytes = read_file()
    
    # Wait until user uploads an image.
    if image_bytes != None:
        predict_and_show(image_bytes, snow_cutoff)
    
if __name__ == '__main__':
	main()
