import joblib

model_path = '/content/trained_model.pkl'  # Path to the pickle file in your Google Drive
model = joblib.load(model_path)


import streamlit as st
from PIL import Image
import numpy as np
import cv2
from google.colab.patches import cv2_imshow


def main():
    # Give your app a title
    st.title("Mask Classifier")

    # Add a file uploader component
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        uploaded_image = Image.open(uploaded_file)
        
        image_array = np.array(uploaded_image)  

        input_image_resized = cv2.resize(image_array, (128,128))

        input_image_scaled = input_image_resized/255

        input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

        input_prediction = model.predict(input_image_reshaped)

        st.write(input_prediction)


        input_pred_label = np.argmax(input_prediction)

        st.write(input_pred_label)


        if input_pred_label == 1:

          st.write('The person in the image is wearing a mask')

        else:

          st.write('The person in the image is not wearing a mask')
              
        

if __name__ == '__main__':
    main()