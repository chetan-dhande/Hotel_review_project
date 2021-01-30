
"""
Created on Fri Jan  1 12:23:39 2021

@author: Chetan
"""

import pandas as pd
import numpy as np 
import streamlit as st 
from pickle import load
from nltk.stem.wordnet import WordNetLemmatizer
import re

from PIL import Image 
img = Image.open("logo.jpeg")
st.image(img) 


html_temp = """ 

<head>
<style>
body {
  background-image: url('https://thumbs.dreamstime.com/b/abstract-blur-hotel-lobby-room-interior-background-vintage-light-filter-71895948.jpg');
  background-repeat: no-repeat;
  background-attachment: fixed;
  background-size: 100% 100%;
}
</style>

</head>

<body bgcolor=#ADD8E6>
 <div class="login">
	<center><h1>Hotel Reviews Analysis</h1></center>


    """

st.markdown(html_temp, unsafe_allow_html = True ) 

review = st.text_area("Review","type here")
data = review.title()


data = (re.sub('[^a-zA-Z]', ' ', str(data)))
data = str(data).lower()
text = (data.split(" "))
df =[]
for i in text:
    df.append(WordNetLemmatizer().lemmatize(i,"v"))

data = " ".join(df)


loaded_model = load(open('model.sav', 'rb'))
output = loaded_model.predict([data])

if(st.button('Submit')): 
    if output==0:
        result = "Negative Review" 
    elif output == 1:
        result = "Neutral Review"
    else:
        result = "Positive Review"
    st.error(result) 



