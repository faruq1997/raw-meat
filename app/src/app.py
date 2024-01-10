import streamlit as st
import tensorflow as tf
import joblib
import cv2
import numpy as np
from sklearn.metrics import accuracy_score

# Load models
resnet_model = tf.keras.models.load_model('./resnet.h5')
gradientboost_model = joblib.load('./gradientboost.pkl')

# Function to preprocess image for ResNet model
def preprocess_image_resnet(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to preprocess image for Gradient Boosting model (GLCM extraction)
def preprocess_image_gb(image):
    img = cv2.imread(image)
    # Perform GLCM extraction and other necessary preprocessing steps
    # Replace this with your GLCM extraction and preprocessing logic
    return img

# Function for classification
def classify_image(image):
    resnet_input = preprocess_image_resnet(image)
    resnet_prediction = resnet_model.predict(resnet_input)
    
    gb_input = preprocess_image_gb(image)
    gb_prediction = gradientboost_model.predict(gb_input)
    
    return resnet_prediction, gb_prediction

# Streamlit app
st.title('Red Meat Classification')
st.write('Upload an image of red meat (pig, goat, cow) for classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    resnet_pred, gb_pred = classify_image(uploaded_file)

    # Display ResNet prediction and accuracy
    st.write("ResNet Model Prediction:")
    resnet_class = np.argmax(resnet_pred)
    resnet_labels = {0: 'Pig', 1: 'Goat', 2: 'Cow'}
    st.write(f"Predicted Class: {resnet_labels[resnet_class]}")
    st.write(f"Prediction Probabilities: {resnet_pred.squeeze()}")
    st.write("")

    # Display Gradient Boosting prediction and accuracy
    st.write("Gradient Boosting Model Prediction:")
    gb_class = np.argmax(gb_pred)
    gb_labels = {0: 'Pig', 1: 'Goat', 2: 'Cow'}
    st.write(f"Predicted Class: {gb_labels[gb_class]}")
    st.write(f"Prediction Probabilities: {gb_pred}")
    st.write("")

    # Calculate and display average accuracy
    resnet_accuracy = accuracy_score([resnet_class], [true_label])  # Replace true_label with the actual label
    gb_accuracy = accuracy_score([gb_class], [true_label])  # Replace true_label with the actual label
    average_accuracy = (resnet_accuracy + gb_accuracy) / 2
    st.write(f"Average Accuracy: {average_accuracy}")
