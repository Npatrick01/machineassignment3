# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:43:41 2020

@author: Admin 2
"""
import streamlit as st 
import tensorflow as tf
import joblib,os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

model=tf.keras.models.load_model( 'model.h5' )

st.title("News Classifier")
	# st.subheader("ML App with Streamlit")
html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">Streamlit ML App </h1>
	</div>

	"""
st.markdown(html_temp,unsafe_allow_html=True)
    
news_text = st.text_area("Enter News Here","Type Here")
if st.button("Classify"):
    vocab_size = 50
    encoded_docs = [one_hot(d, vocab_size) for d in news_text]
    max_length = 10
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding= 'post' )
    y_pred = model.predict(padded_docs)
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    y_pred*100

    
    # load the model
    st.write(np.argmax(y_pred[0]))  
    if(y_pred[0]==1):
        st.write('sparm')
    else:
        st.write('Ham')
        






