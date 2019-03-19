
# coding: utf-8

# In[1]:


#importing the important modules
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import tensorflow as tf
import keras


# In[2]:


#loading all the files in the form of data frames
facebook_economy = pd.read_csv("Facebook_Economy.csv")
facebook_microsoft = pd.read_csv("Facebook_Microsoft.csv")
facebook_obama = pd.read_csv("Facebook_Obama.csv")
facebook_palestine = pd.read_csv("Facebook_Palestine.csv")
googleplus_economy = pd.read_csv("GooglePlus_Economy.csv")
googleplus_microsoft = pd.read_csv("GooglePlus_Microsoft.csv")
googleplus_obama = pd.read_csv("GooglePlus_Obama.csv")
googleplus_palestine = pd.read_csv("GooglePlus_Palestine.csv")
linkedin_economy = pd.read_csv("LinkedIn_Economy.csv")
linkedin_microsoft = pd.read_csv("LinkedIn_Microsoft.csv")
linkedin_obama = pd.read_csv("LinkedIn_Obama.csv")
linkedin_palestine = pd.read_csv("LinkedIn_Palestine.csv")

news_final=pd.read_csv("News_Final.csv")

news_final.describe()
print(news_final.shape)


# In[3]:


#drop all the rows that have value (-1,-1,-1) for facebook,google+ and linked in
news_final[["Facebook","GooglePlus","LinkedIn"]] = news_final[["Facebook","GooglePlus","LinkedIn"]] .replace(-1, np.NaN)

news_final.dropna(inplace=True)
#to check the number of rows that are left now
print(news_final.shape)


# In[4]:


#Data cleaning
#removing unnecessary punctuations and tags

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    
    return input_txt

""" What vectorize does is it takes a nested sequence of objects or numpy arrays as inputs 
returns a single or tuple of numpy array as output.
"""
news_final['Title'] = np.vectorize(remove_pattern)(news_final['Title'], "@[\\w]*")

#replacing numbers and other special characters with a space in the Title elements
news_final['Title'] = news_final['Title'].str.replace("[^a-zA-Z#]", " ")

news_final['Headline'] = np.vectorize(remove_pattern)(news_final['Headline'], "@[\\w]*")

#replacing numbers and other special characters with a space in the Title elements
news_final['Headline'] = news_final['Headline'].str.replace("[^a-zA-Z#]", " ")

#remove stopwords
stop = stopwords.words('english')
news_final= pd.DataFrame(news_final)
news_final['Title'] = news_final['Title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


stop = stopwords.words('english')
news_final= pd.DataFrame(news_final)
news_final['Headline'] = news_final['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[5]:


news_final.dtypes


# In[6]:


#we are creating a list with zipped values of id,title,headline,source,topic and number of shares for Facebook
z=[]
z=list(zip(news_final['IDLink'],news_final['Title'],news_final['Headline'],news_final['Source'],news_final['Topic'],news_final['Facebook']))

#sorting the list in ascending order according to number of shares
z1=sorted(z, key=lambda t: t[5])

#creating a dataframe to store the list elements into the dataframe
labels = ['IDLink','Title', 'Headline','Source', 'Topic', 'SharesOfFacebook']
df_facebook = pd.DataFrame.from_records(z1, columns=labels)

#dividing the shares in categories to improve the accuracy of the algorithm
def label_emotion (row):
   if row['SharesOfFacebook'] >=0 and row['SharesOfFacebook'] <=12500  :
      return 1
   if row['SharesOfFacebook'] > 12500 and row['SharesOfFacebook']<=25000:
      return 2
   if row['SharesOfFacebook'] > 25000 and row['SharesOfFacebook']<=37500:
      return 3
   if row['SharesOfFacebook'] >37500 and row['SharesOfFacebook'] <=50000:
      return 4
df_facebook['category_facebook'] = df_facebook.apply (lambda row: label_emotion (row),axis=1)

df_facebook


# In[7]:


#we are creating a list with zipped values of id,title,headline,source,topic and number of shares for GooglePlus
z=[]
z=list(zip(news_final['IDLink'],news_final['Title'],news_final['Headline'],news_final['Source'],news_final['Topic'],news_final['GooglePlus']))

#sorting the list in ascending order according to number of shares
z1=sorted(z, key=lambda t: t[5])

#creating a dataframe to store the list elements into the dataframe
labels = ['IDLink','Title', 'Headline','Source', 'Topic', 'SharesOfGooglePlus']
df_googlePlus = pd.DataFrame.from_records(z1, columns=labels)

#dividing the shares in categories to improve the accuracy of the algorithm
def label_emotion (row):
   if row['SharesOfGooglePlus'] >=0 and row['SharesOfGooglePlus'] <=65  :
      return 1
   if row['SharesOfGooglePlus'] > 65 and row['SharesOfGooglePlus']<=130:
      return 2
   if row['SharesOfGooglePlus'] > 130 and row['SharesOfGooglePlus']<=195:
      return 3
   if row['SharesOfGooglePlus'] >195 and row['SharesOfGooglePlus'] <=260:
      return 4
   if row['SharesOfGooglePlus'] >260 and row['SharesOfGooglePlus'] <=325:
      return 5
   if row['SharesOfGooglePlus'] >325 and row['SharesOfGooglePlus'] <=390:
      return 6
   if row['SharesOfGooglePlus'] >390 and row['SharesOfGooglePlus'] <=455:
      return 7
   if row['SharesOfGooglePlus'] >455 and row['SharesOfGooglePlus'] <=520:
      return 8
   if row['SharesOfGooglePlus'] >520 and row['SharesOfGooglePlus'] <=585:
      return 9
   if row['SharesOfGooglePlus'] >585 and row['SharesOfGooglePlus'] <=650:
      return 10
   if row['SharesOfGooglePlus'] >=650 and row['SharesOfGooglePlus'] <=715  :
      return 11
   if row['SharesOfGooglePlus'] > 715 and row['SharesOfGooglePlus']<=780:
      return 12
   if row['SharesOfGooglePlus'] > 780 and row['SharesOfGooglePlus']<=845:
      return 13
   if row['SharesOfGooglePlus'] >845 and row['SharesOfGooglePlus'] <=910:
      return 14
   if row['SharesOfGooglePlus'] >910 and row['SharesOfGooglePlus'] <=975:
      return 15
   if row['SharesOfGooglePlus'] >975 and row['SharesOfGooglePlus'] <=1040:
      return 16
   if row['SharesOfGooglePlus'] >1040 and row['SharesOfGooglePlus'] <=1105:
      return 17
   if row['SharesOfGooglePlus'] >1105 and row['SharesOfGooglePlus'] <=1170:
      return 18
   if row['SharesOfGooglePlus'] >1170 and row['SharesOfGooglePlus'] <=1235:
      return 19
   if row['SharesOfGooglePlus'] >1235 and row['SharesOfGooglePlus'] <=1300:
      return 20
df_googlePlus['category_GooglePlus'] = df_googlePlus.apply (lambda row: label_emotion (row),axis=1)

df_googlePlus

#on dividing into more number of categories,it gives better accuracy 99.6(ten categories) 99.8(twenty categories)


# In[8]:


#we are creating a list with zipped values of id,title,headline,source,topic and number of shares for LinkedIn
z=[]
z=list(zip(news_final['IDLink'],news_final['Title'],news_final['Headline'],news_final['Source'],news_final['Topic'],news_final['LinkedIn']))

#sorting the list in ascending order according to number of shares
z1=sorted(z, key=lambda t: t[5])

#creating a dataframe to store the list elements into the dataframe
labels = ['IDLink','Title', 'Headline','Source', 'Topic', 'SharesOfLinkedIn']
df_linkedIn = pd.DataFrame.from_records(z1, columns=labels)

#dividing the shares in categories to improve the accuracy of the algorithm
def label_emotion (row):
   if row['SharesOfLinkedIn'] >=0 and row['SharesOfLinkedIn'] <=1600  :
      return 1
   if row['SharesOfLinkedIn'] > 1600 and row['SharesOfLinkedIn']<=3200:
      return 2
   if row['SharesOfLinkedIn'] > 3200 and row['SharesOfLinkedIn']<=4800:
      return 3
   if row['SharesOfLinkedIn'] >4800 and row['SharesOfLinkedIn'] <=6400:
      return 4
df_linkedIn['category_LinkedIn'] = df_linkedIn.apply (lambda row: label_emotion (row),axis=1)

df_linkedIn


# In[11]:


"""Keras provides the one_hot() function that creates a hash of each word as an efficient integer encoding.
We will estimate the vocabulary size of 50, 
which is much larger than needed to reduce the probability of collisions from the hash function.
"""
from keras.preprocessing.text import one_hot#import has not been moved from here for better understandability of the code
# integer encode the documents
vocab_size = 100
encoded_docs = [one_hot(d, vocab_size) for d in df_facebook['Title']]


"""The sequences have different lengths and Keras prefers inputs to be vectorized and all inputs to have the same length. 
We will pad all input sequences to have the length of 4.
Again, we can do this with a built in Keras function, in this case the pad_sequences() function.
pad documents to a max length of 4 words
"""
from keras.preprocessing.sequence import pad_sequences
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')



"""We are now ready to define our Embedding layer as part of our neural network model.
The Embedding has a vocabulary of 50 and an input length of 4. We will choose a small embedding space of 8 dimensions.
The model is a simple binary classification model. 
Importantly, the output from the Embedding layer will be 4 vectors of 8 dimensions each, one for each word. 
We flatten this to a one 32-element vector to pass on to the Dense output layer.
define the model
"""
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Flatten
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])#loss='binary_crossentropy'


#Finally, we can fit and evaluate the classification model.
# fit the model
model.fit(padded_docs, df_facebook['category_facebook'], epochs=20, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, df_facebook['category_facebook'], verbose=0)
print('Accuracy: %f' % (accuracy*100))


# In[12]:


"""Keras provides the one_hot() function that creates a hash of each word as an efficient integer encoding.
We will estimate the vocabulary size of 50, 
which is much larger than needed to reduce the probability of collisions from the hash function.
"""
from keras.preprocessing.text import one_hot#import has not been moved from here for better understandability of the code
# integer encode the documents
vocab_size = 100
encoded_docs = [one_hot(d, vocab_size) for d in df_googlePlus['Title']]


"""The sequences have different lengths and Keras prefers inputs to be vectorized and all inputs to have the same length. 
We will pad all input sequences to have the length of 4.
Again, we can do this with a built in Keras function, in this case the pad_sequences() function.
 pad documents to a max length of 4 words
 """
from keras.preprocessing.sequence import pad_sequences
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


"""We are now ready to define our Embedding layer as part of our neural network model.
The Embedding has a vocabulary of 50 and an input length of 4. We will choose a small embedding space of 8 dimensions.
The model is a simple binary classification model. 
Importantly, the output from the Embedding layer will be 4 vectors of 8 dimensions each, one for each word. 
We flatten this to a one 32-element vector to pass on to the Dense output layer.
define the model
"""
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Flatten
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])#loss='binary_crossentropy'


#Finally, we can fit and evaluate the classification model.
# fit the model
model.fit(padded_docs, df_googlePlus['category_GooglePlus'], epochs=20, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, df_googlePlus['category_GooglePlus'], verbose=0)
print('Accuracy: %f' % (accuracy*100))


# In[13]:


"""Keras provides the one_hot() function that creates a hash of each word as an efficient integer encoding.
We will estimate the vocabulary size of 50, 
which is much larger than needed to reduce the probability of collisions from the hash function.
"""

from keras.preprocessing.text import one_hot#import has not been moved from here for better understandability of the code
# integer encode the documents
vocab_size = 100
encoded_docs = [one_hot(d, vocab_size) for d in df_linkedIn['Title']]


"""The sequences have different lengths and Keras prefers inputs to be vectorized and all inputs to have the same length. 
We will pad all input sequences to have the length of 4.
Again, we can do this with a built in Keras function, in this case the pad_sequences() function.
pad documents to a max length of 4 words
"""
from keras.preprocessing.sequence import pad_sequences
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


"""We are now ready to define our Embedding layer as part of our neural network model.
The Embedding has a vocabulary of 50 and an input length of 4. We will choose a small embedding space of 8 dimensions.
The model is a simple binary classification model. 
Importantly, the output from the Embedding layer will be 4 vectors of 8 dimensions each, one for each word. 
We flatten this to a one 32-element vector to pass on to the Dense output layer.
define the model
"""
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Flatten
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])#loss='binary_crossentropy'

#Finally, we can fit and evaluate the classification model.
# fit the model
model.fit(padded_docs, df_linkedIn['category_LinkedIn'], epochs=20, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, df_linkedIn['category_LinkedIn'], verbose=0)
print('Accuracy: %f' % (accuracy*100))


# In[14]:


# create the transform
vectorizer = HashingVectorizer(n_features=20)
# encode document
vector = vectorizer.transform(df_facebook['Title'])
vec=vector.toarray()

#Split the data into training and test sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X = vec
y = df_facebook.iloc[:,6]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#Fit logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


#Predicting the test set results and creating confusion matrix
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


"""Compute precision, recall, F-measure and support
The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.
The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
The recall is intuitively the ability of the classifier to find all the positive samples.
The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
The F-beta score weights the recall more than the precision by a factor of beta. beta = 1.0 means recall and precision are equally important.
The support is the number of occurrences of each class in y_test.
"""
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[16]:


# create the transform
vectorizer = HashingVectorizer(n_features=20)
# encode document
vector = vectorizer.transform(df_googlePlus['Title'])
vec=vector.toarray()

#Split the data into training and test sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X = vec
y = df_googlePlus.iloc[:,6]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Fit logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


#Predicting the test set results and creating confusion matrix
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


"""Compute precision, recall, F-measure and support
The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.
The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
The recall is intuitively the ability of the classifier to find all the positive samples.
The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
The F-beta score weights the recall more than the precision by a factor of beta. beta = 1.0 means recall and precision are equally important.
The support is the number of occurrences of each class in y_test.
"""
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[17]:


# create the transform
vectorizer = HashingVectorizer(n_features=20)
# encode document
vector = vectorizer.transform(df_linkedIn['Title'])
vec=vector.toarray()

#Split the data into training and test sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X = vec
y = df_linkedIn.iloc[:,6]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Fit logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Predicting the test set results and creating confusion matrix
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

"""Compute precision, recall, F-measure and support
The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.
The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
The recall is intuitively the ability of the classifier to find all the positive samples.
The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
The F-beta score weights the recall more than the precision by a factor of beta. beta = 1.0 means recall and precision are equally important.
The support is the number of occurrences of each class in y_test.
"""
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

