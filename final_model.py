
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
from pickle import dump
import re

#df1 = pd.read_csv("model_1.csv")
df3 = pd.read_csv("rating.csv")
df2 = pd.read_csv("model_2.csv")
#print(df1.shape)
print(df2.shape)
print(df3.shape)
data = pd.concat([df2,df3],axis=1)
print(data.Rating.value_counts())

def posneg(x):
  if x<3:
    return 0 
  elif x==3:
    return 1
  else:  
    return 2   

data["rating"] = data["Rating"].apply(lambda x : posneg(x))
print("Before")
print(data["rating"].value_counts())

X = data["text"]
y = data["rating"]
X_smote,X_test,y_smote,y_test = train_test_split(X,y,test_size=0.35,random_state=20)

X_smote = X_smote.values.reshape(-1,1)
y_smote = y_smote.values.reshape(-1,1)

smote = RandomOverSampler()
train,y_train = smote.fit_sample(X_smote,y_smote)

X_train = []
for i in train:
  X_train.append(str(i))

df = pd.concat([pd.DataFrame(X_train),pd.DataFrame(y_train)],axis=1)
df.columns = ["text","rating"]
a = df[df["rating"]==0].sample(6000)
b = df[df["rating"]==1]
c = df[df["rating"]==2].sample(5000)
df1 = pd.concat([a,b,c],axis=0)

X_train = df1["text"]
y_train = df1["rating"]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

"""# Logistic """

pipline_2 = Pipeline([("tfidf",TfidfVectorizer()),
                    ("model",LogisticRegression(max_iter=400))])
pipline_2.fit(X_train,y_train)
print("Logistic")
print("training_accuracy: ",pipline_2.score(X_train,y_train))
print("testing_accuracy: ",pipline_2.score(X_test,y_test))
print("classification_report")
print(classification_report(y_test,pipline_2.predict(X_test)))
print("Cross_val_score_train")
print(cross_val_score(pipline_2,X_train,y_train,cv=3))
print("Cross_val_score_test")
print(cross_val_score(pipline_2,X_test,y_test,cv=3))

print("confusion_matrix_test")
print(confusion_matrix(y_test,pipline_2.predict(X_test)))

print("confusion_matrix_train")
print(confusion_matrix(y_train,pipline_2.predict(X_train)))

def model_1(review):
  prediction = pipline_2.predict(review)
  return prediction

abc  =["nice expensive park good deal stay anniversary  arrive late even advice previous review valet park  check quick easy  disappoint existent view clean nice size  bed comfortable wake stiff neck high pillow  not soundproof hear music morning loud bang doors open close hear people talk hallway  noisy neighbor  aveda bath products nice  not goldfish stay nice touch advantage stay longer #$%^&* DFGGT Location Great <?,7474544654635 location great walk distance shop  nice experience pay    park  "]

model_1(abc)

text = (re.sub('[^a-zA-Z]', ' ', abc))
text = (text).lower()

print(df2["text"][5])

dump(pipline_2,open('model.sav','wb'))