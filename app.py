import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
st.title('News_Classifier ML App')
st.subheader('NLP and ML App with Streamlit')
activites=['News_Classifier']
choice=st.sidebar.selectbox('Choose Activites',activites)
if choice=='News_Classifier':
  all_models=['NN']
  class_names={'Business':0,'Tech':1,'Politics':2,'Sport':3,'Entertainment':4}
  news_text=st.text_area('Enter text','Type here.....')
  model_choice=st.selectbox('Choice model',all_models)
  if st.button('Predict'):
    st.text('Original Text:\n{}'.format(news_text))
  def get_value(val,my_dict):
    for a,b in my_dict.items():
      if b==val:
        value=a
    return value
  if model_choice=='NN':
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    with open('tokenizer.pickle','rb') as handle:
      vec=pickle.load(handle)
    def predictions(model,texts):
      data=vec.texts_to_sequences([texts])
      pad=pad_sequences(data,maxlen=240)
      pred=model.predict(pad)
      pred=np.argmax(pred,axis=1)
      y_pred=get_value(pred,class_names)
      return y_pred
    if news_text is None:
      st.text('Provide your news')
    else:
      predictions=predictions(model,news_text)
      st.success('News Categorized as {}'.format(predictions))
