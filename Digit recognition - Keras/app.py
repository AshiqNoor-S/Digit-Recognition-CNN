import streamlit as st
import numpy as np
from PIL import Image
import keras
from keras.preprocessing import image as keras_image

# Load your trained model
model_path = 'convolutional_model.h5'  # Update the path as needed
model = keras.models.load_model(model_path)

# Streamlit setup
st.title('Convolutional Neural Network App')
st.write('Interact with the CNN model for digit recognition!')

st.sidebar.info(
    "This app uses a Convolutional Neural Network to recognize handwritten digits."
)

st.sidebar.title("Instructions")
st.sidebar.write("1. Upload an image of a handwritten digit (0-9).")
st.sidebar.write("2. Click the 'Predict' button to see the model's prediction.")

st.sidebar.write("Pls note : As this model is trained using 28x28 images , the image uploaded will also be downgraded to this resolution , so kindly upload proper images of digits , which can stay intact even after compression")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    resized_image = image.resize((28, 28))
    grayscale_image = resized_image.convert('L')  # Convert to grayscale
     
    st.write('Since the model is trained with 28x28 gray scale images , the uploaded image is also converted to similar format !')
    st.image(grayscale_image, caption='Processed image', use_column_width=True)

    image_array = np.array(grayscale_image)  # Convert to NumPy array
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Debugging output
    st.write("Preprocessed Image Shape:", image_array.shape)
    st.write("Preprocessed Image Min:", np.min(image_array))
    st.write("Preprocessed Image Max:", np.max(image_array))

    if st.button('Predict'):
        prediction = model.predict(image_array)
        st.write("Raw Prediction:", prediction)
        digit = np.argmax(prediction)
        confidence = prediction[0][digit]

        st.write(f"Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence:.2f}")
