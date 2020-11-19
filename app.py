import streamlit as st
import numpy as np
import joblib
import spacy
import pickle
import tensorflow as tf
import pandas as pd
from wordcloud import WordCloud
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
nlp=spacy.load('en')

vectorizer_file=open('vectorizer.pkl','rb')
vec=joblib.load(vectorizer_file)

st.title('News_Classifier ML App')
st.subheader('NLP and ML App with Streamlit')
activites=['News_Classifier','NLP_tokenization']
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
    model=tf.keras.models.load_model('tfmodel.h5')
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
if choice=='NLP_tokenization':
  st.info('Natural Language Processing')
  news_text=st.text_area('Enter Text','Type Here')
  nlp_task=['Tokenizer','Lemmatization','Pos_tags']
  task_choice=st.selectbox('Choose NLP task',nlp_task)
  if st.button('Analyze'):
    st.info('Original text: {}'.format(news_text))
    docx=nlp(news_text)
    if task_choice=='Tokenizer':
      result=[token.text for token in docx]
    elif task_choice=='Lemmatization':
      result=['Tokenized text:{}, Lemmatized text:{}'.format(lemma.text,lemma.lemma_) for lemma in docx]
    elif task_choice=='Pos_tags':
      result=['Token:{},Pos_tags:{},Dependency:{}'.format(word.text,word.tag_,word.dep_) for word in docx]
    st.json(result)
  if st.button('Tabulize'):
    docx=nlp(news_text)
    c_tokens=[token.text for token in docx]
    c_lemma=[lemma.lemma_ for lemma in docx]
    c_pos=[(word.text,word.tag_,word.dep_) for word in docx]
    new_df=pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','Pos'])
    st.dataframe(new_df)
  if st.checkbox('WordCloud'):
    plt.figure(figsize=(20,20))
    wordcloud=WordCloud().generate(news_text)
    plt.imshow(wordcloud)
    st.pyplot()
