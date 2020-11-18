{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled1.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyNONMOPsKZ/bXQgrmHT8bVD",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Riaduddin/bbc_news/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5btA4ElJJf_z",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "de69dd16-3ecd-4be1-bc89-fa31f7f63b3b"
      },
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "import numpy as np\n",
        "import joblib,os\n",
        "import spacy\n",
        "import pickle\n",
        "import tensorflow as tf\n",
        "import pandas as pd\n",
        "from wordcloud import WordCloud\n",
        "from PIL import Image\n",
        "from tensorflow.keras.preprocessing.text import Tokenizer\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "import matplotlib.pyplot as plt\n",
        "nlp=spacy.load('en')\n",
        "\n",
        "vectorizer_file=open('/content/vectorizer.pkl','rb')\n",
        "vec=joblib.load(vectorizer_file)\n",
        "\n",
        "st.title('News_Classifier ML App')\n",
        "st.subheader('NLP and ML App with Streamlit')\n",
        "activites=['News_Classifier','NLP_tokenization']\n",
        "choice=st.sidebar.selectbox('Choose Activites',activites)\n",
        "if choice=='News_Classifier':\n",
        "  all_models=['NN']\n",
        "  class_names={'Business':0,'Tech':1,'Politics':2,'Sport':3,'Entertainment':4}\n",
        "  news_text=st.text_area('Enter text','Type here.....')\n",
        "  model_choice=st.selectbox('Choice model',all_models)\n",
        "  if st.button('Predict'):\n",
        "    st.text('Original Text:\\n{}'.format(news_text))\n",
        "  def get_value(val,my_dict):\n",
        "    for a,b in my_dict.items():\n",
        "      if b==val:\n",
        "        value=a\n",
        "    return value\n",
        "  if model_choice=='NN':\n",
        "    model=tf.keras.models.load_model('/content/tfmodel.h5')\n",
        "    with open('/content/tokenizer.pickle','rb') as handle:\n",
        "      vec=pickle.load(handle)\n",
        "    def predictions(model,texts):\n",
        "      data=vec.texts_to_sequences([texts])\n",
        "      pad=pad_sequences(data,maxlen=240)\n",
        "      pred=model.predict(pad)\n",
        "      pred=np.argmax(pred,axis=1)\n",
        "      y_pred=get_value(pred,class_names)\n",
        "      return y_pred\n",
        "    if news_text is None:\n",
        "      st.text('Provide your news')\n",
        "    else:\n",
        "      predictions=predictions(model,news_text)\n",
        "      st.success('News Categorized as {}'.format(predictions))\n",
        "if choice=='NLP_tokenization':\n",
        "  st.info('Natural Language Processing')\n",
        "  news_text=st.text_area('Enter Text','Type Here')\n",
        "  nlp_task=['Tokenizer','Lemmatization','Pos_tags']\n",
        "  task_choice=st.selectbox('Choose NLP task',nlp_task)\n",
        "  if st.button('Analyze'):\n",
        "    st.info('Original text: {}'.format(news_text))\n",
        "    docx=nlp(news_text)\n",
        "    if task_choice=='Tokenizer':\n",
        "      result=[token.text for token in docx]\n",
        "    elif task_choice=='Lemmatization':\n",
        "      result=['Tokenized text:{}, Lemmatized text:{}'.format(lemma.text,lemma.lemma_) for lemma in docx]\n",
        "    elif task_choice=='Pos_tags':\n",
        "      result=['Token:{},Pos_tags:{},Dependency:{}'.format(word.text,word.tag_,word.dep_) for word in docx]\n",
        "    st.json(result)\n",
        "  if st.button('Tabulize'):\n",
        "    docx=nlp(news_text)\n",
        "    c_tokens=[token.text for token in docx]\n",
        "    c_lemma=[lemma.lemma_ for lemma in docx]\n",
        "    c_pos=[(word.text,word.tag_,word.dep_) for word in docx]\n",
        "    new_df=pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','Pos'])\n",
        "    st.dataframe(new_df)\n",
        "  if st.checkbox('WordCloud'):\n",
        "    plt.figure(figsize=(20,20))\n",
        "    wordcloud=WordCloud().generate(news_text)\n",
        "    plt.imshow(wordcloud)\n",
        "    st.pyplot()\n",
        "\n",
        "\n",
        "\n"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Overwriting app.py\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kNL2eokTYWkL"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}