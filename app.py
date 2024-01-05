import streamlit as st
import tensorflow as tf
from utils import clean_tweet,response

model = tf.keras.models.load_model("hate_speech_detection_model")

st.title("Hate Speech and Offensive Language Detections")
input_text = st.text_input("Enter The Tweet")
pred = st.button("Predict")
st.divider()
if pred:
    cleaned_test = clean_tweet(input_text)
    response_text = model.predict([cleaned_test])
    st.text_area('Response:- ',response(response_text))
