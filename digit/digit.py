#import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import pickle

st.header("Machine Learning Handwritten Digit Images for prediction")
Upload_image_file = st.file_uploader("Upload a Handwritten ms paint image")
if Upload_image_file !=None:
    opening_img = Image.open(Upload_image_file,'r')
    Image_1 = st.image(opening_img)
    resize_img = opening_img.resize((28,28),Image.ANTIALIAS)
    resized_arr_image = np.array(resize_img)
    resized_arr_image = resized_arr_image[:,:,2]
    img_flattened = resized_arr_image.flatten()
    final_image = img_flattened.reshape(1,-1)
    st.write(final_image)
    if st.button('predict'):
        Lg_classifier = pickle.load(open(r'C:\Users\Lakshman\Downloads\Digit_model\logreg_digit_model.pkl',"rb"))
        y_pred = Lg_classifier.predict(final_image)
        st.header("The uploaded image contains the digit :")
        st.title(y_pred)
    else:
        st.write('Press predict button')
else:
    st.write("Upload an image")