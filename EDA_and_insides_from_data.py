# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:58:52 2021

@author: Chetan
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:07:41 2020

@author: Chetan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer 
import seaborn as sns
from nltk.stem.porter import PorterStemmer
data = pd.read_csv("C://Users//Chetan//Desktop//Project//text.csv")


rating = pd.DataFrame(data.rating.value_counts()).reset_index()
type(rating)
rating.columns
rating.columns = ['number','rating']
sns.barplot(rating.number,rating.rating,order=[5,4,3,2,1])

data.columns
data.text[0]
data.text[1]

sample =data.text[1]


with open("D:\\chetan\\assignment\\14text mining\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]

with open("D:\\chetan\\assignment\\14text mining\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

sample = sample.split(" ")
ip_neg_in_neg = " ".join([w for w in sample if w in negwords])

word = negwords
for i in poswords:
    word.append(i)

df = []

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
ip_pos_in_pos = " ".join ([w for w in data.text if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)


negative = pd.DataFrame(ip_neg_in_neg.split(" "))
positive= pd.DataFrame(ip_pos_in_pos.split(" "))

negative.count()
positive.count()


positivity = (positive.count()/word.count())*100#8.85%
negativity = (negative.count()/word.count())*100#5.05%
neutral = 100-(positivity+negativity)#86.085%


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:15:18 2020

@author: Chetan
"""
#importing necessary library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer 
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


### import data
data = pd.read_csv("C://Users//Chetan//Desktop//Project//dataset//hotel_reviews.csv")

## removing unneccesary word,symbol
text=[]
for i in data["Review"]:
    y = (re.sub('[^a-zA-Z]', ' ', i))
    y = y.lower()
    text.append(y)
data["text"] = text

##importing posiitive and negative word 

with open("D:\\chetan\\assignment\\14text mining\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
poswords = poswords[36:] 
with open("D:\\chetan\\assignment\\14text mining\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")
negwords = negwords[37:]

stop = stopwords.words('english')
## to remove stop word and lemitization 
def lem_word(x):
    stop = stopwords.words('english')
    text = list(x.split(" "))
    text =[WordNetLemmatizer().lemmatize(w,"v") for w in text if w not in stop]
    return text
    
clean_text = data["text"].apply(lambda x: lem_word(x))
data["clean_text"] = clean_text

# positive word extraction
def positive(x):
    text = [w for w in x if w in poswords]
    return text

positive_word =  clean_text.apply(lambda x: positive(x))

#negative word extraction                                                        
def negative(x):
    text = [w for w in x if w in negwords]
    return text

negative_word =  clean_text.apply(lambda x: negative(x))

# to remove unnecessary spaces 
def clr(x):
    a = []
    for i in x:
        if len(i)>1:
            a.append(i)
    return a
            
negative_word = negative_word.apply(lambda x: clr(x))
positive_word = positive_word.apply(lambda x: clr(x))
clean_text = clean_text.apply(lambda x: clr(x))

data["clean_text"] = clean_text
data["positive_word"] = positive_word
data["negative_word"] = negative_word

data["text"] = clean_text.apply(lambda x:" ".join(x))


data["clean_text_count"] = data["clean_text"].apply(lambda x:len(x))
data["positive_word_count"] = data["positive_word"].apply(lambda x:len(x))
data["negative_word_count"] = data["negative_word"].apply(lambda x:len(x))

data["positive_percentage"] = (data["positive_word_count"]/data["clean_text_count"])*100
data["negative_percentage"] = (data["negative_word_count"]/data["clean_text_count"])*100

data.to_csv("text.csv")



# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:38:53 2020

@author: Chetan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer 
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

data = pd.read_csv("C://Users//Chetan//Desktop//Project//text.csv")

x = data["clean_text_count"].median()
x_neg = data["negative_word_count"].median()
x_pos = data["positive_word_count"].median()

## percentage of text pie chart
cat = [x,x_neg,x_pos]
wedgeprops = {"linewidth":3,"width":1,"edgecolor":"k"}
plt.pie(cat,labels=["normal text","negativity","positivity"],autopct="%0.1f%%",
        explode=[0.1,0.2,0.1],textprops={"fontsize":12})

## rating bargraph

rating = pd.DataFrame(data.Rating.value_counts()).reset_index()
rating.columns = ['number','rating']
sns.barplot(rating.number,rating.rating,order=[5,4,3,2,1])

### text count histogram
data = pd.read_csv("text.csv")

sns.distplot(data["clean_text_count"],bins=2000,axlabel="avrage Review word count")

sns.distplot(data["positive_word_count"],bins=200,axlabel="avrage positive word count")

sns.distplot(data["negative_word_count"],bins=150,axlabel="avrage negative word count")

x = ["Clean_text",'positive_word',"negative word"]
y = [data["clean_text_count"].median(),data["positive_word_count"].median(),
     data["negative_word_count"].median()]
sns.barplot(x,y)

data["clean_text_count"].min()
data["clean_text_count"].max()
data["clean_text_count"].median()
data["clean_text_count"].std()

data["positive_word_count"].min()
data["positive_word_count"].max()
data["positive_word_count"].median()
data["positive_word_count"].std()

data["negative_word_count"].min()
data["negative_word_count"].max()
data["negative_word_count"].median()
data["negative_word_count"].std()

low_rate = data[data["Rating"]<3]
high_rate = data[data["Rating"]>3]



service = ["room","sheets","bed","airconditioner","curtain","rude room","air","conditioner",
           "service","view","chair","breakfast",
           "internet","lunch","location","safe","gym","towel","meal","staff","housekeeping"]

# to extract hotel service to find out  suggestion 
    
def Service(x):
    text = [w for w in x if w in service]
    return text
good_service = high_rate["clean_text"].apply(lambda x:Service(x))   

data["polarity"] = data["Review"].apply(lambda x : TextBlob(x).sentiment.polarity)
data["subjectivity"] = data["Review"].apply(lambda x : TextBlob(x).sentiment.subjectivity)


data["clean_text_count"] = data["clean_text"].apply(lambda x: len(TextBlob(x).words))
data["positive_word_count"] = data["positive_word"].apply(lambda x: len(TextBlob(x).words))
data["negative_word_count"] = data["negative_word"].apply(lambda x: len(TextBlob(x).words))

data["positive_percentage"] = (data["positive_word_count"]/data["clean_text_count"])*100
data["negative_percentage"] = (data["negative_word_count"]/data["clean_text_count"])*100

# data.to_csv("text.csv")

text = " ".join(low_rate["text"])
text = TextBlob(text).words
text = [w for w in text if w in service]
text = " ".join(text)

uni = []
for i in service:
    uni.append(text.count(i))

ser_n = pd.DataFrame(service)    
uni= pd.DataFrame(uni)
ser_n =pd.concat([ser_n,uni],axis=1)        


text1 = " ".join(high_rate["text"][0:500])
text1 = TextBlob(text1).words
text1 = [w for w in text1 if w in service]
text1 =" ".join(text1)

uni = []
for i in service:
    uni.append(text1.count(i))
    
ser = pd.DataFrame(service)    
uni= pd.DataFrame(uni)
ser =pd.concat([ser,uni],axis=1)     
ser.columns = ["service","positive","negative"]    

ser["pos"] = (ser["positive"]/(ser["positive"]+ser["negative"]))*100
ser["neg"] = (ser["negative"]/(ser["positive"]+ser["negative"]))*100




# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:05:35 2020

@author: Chetan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer 
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim

data = pd.read_csv("C://Users//Chetan//Desktop//Project//all csv//text.csv")

low_rate = data[data["Rating"]<3]
high_rate = data[data["Rating"]>3]

stoplist = stopwords.words('english')
c_vec = CountVectorizer(stop_words=stoplist,ngram_range=(3,4))
# matrix of ngrams
ngrams = c_vec.fit_transform(low_rate["text"][0:1000])
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})

df_ngram["polarity"] = TextBlob(str(df_ngram["bigram/trigram"])).polarity
neg = df_ngram[df_ngram["polarity"]<0]
df_ngram = df_ngram.drop(["polarity"],axis = 1)


text = low_rate["text"][0:500]
text = [str(text).split()]

dictionary = corpora.Dictionary([str(text).split()])
doc_term_matrix = [dictionary.doc2bow(rev) for rev in text]

# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=10, random_state=100,
                chunksize=1000, passes=50,iterations=100)
lda_model.print_topics()

from gensim.models.coherencemodel import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_model, texts=text, dictionary=dictionary , coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
lda_model.print_topics(num_words=5)


gram =  TextBlob(str(low_rate["text"][0:500])).ngrams(n=3)
print(high_rate.text[1])
