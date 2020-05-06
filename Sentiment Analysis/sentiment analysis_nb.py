#src: https://www.kaggle.com/sandeepbhutani304/sentiment-analysis-using-naive-bayes

import numpy as np
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import re

from sklearn.model_selection import train_test_split

#read the files from the data folder
for dirname, _, filenames in os.walk('../Data/twitter-sentiment-analysis-hatred-speech/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#step 1: read the training and testing input file
#step 2: preprocess the data
#step 3: count total rows for each label
#step 4: if data is imbalanced, balance the data (here upsampling method is used)
#step 5: select the model (here naive bayes)
#step 6: convert the text into numerical data (here countvectorizer is used)
#step 7: divide the training data into train and test
#step 8: fit and transform data into the model from step 5 -- for training data: fit and transform, for testing data: only transform
#step 9: train the model (x_train, y_train)
#step 10: predict test data gathered from step 7, calculate the accuracy
#step 11: predict no_label_test data

#read the train data
train_orig=pd.read_csv("../Data/twitter-sentiment-analysis-hatred-speech/train.csv")

#read the test data
test_nolabel=pd.read_csv("../Data/twitter-sentiment-analysis-hatred-speech/test.csv")

#get the standard stopwords
stop_words = set(stopwords.words('english'))

train = train_orig

def remove_stopwords(line):
    word_tokens = word_tokenize(line)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)

def preprocess(line):
    line = line.lower()  #convert to lowercase
    line = re.sub(r'\d+', '', line)  #remove numbers
    line = line.translate(line.maketrans("","", string.punctuation))  #remove punctuation
#     line = line.translate(None, string.punctuation)  #remove punctuation
    line = remove_stopwords(line)
    return line


for i,line in enumerate(train.tweet):
    train.tweet[i] = preprocess(line)


#after preprocessing, train the data
X_train, X_test, y_train, y_test = train_test_split(train['tweet'], train['label'], test_size=0.5,
                                                        stratify=train['label'])

trainp = train[train.label == 1]
trainn = train[train.label == 0]

# print(trainp.info()) #2242 rows
# print(trainn.info()) #29720 rows

#balance the dataset
train_imbalanced = train
from sklearn.utils import resample

df_majority = train[train.label == 0]
df_minority = train[train.label == 1]

# Upsample minority class
# upsample: for every observation in the majority class, we randomly select an observation from the minority class with replacement.
# The end result is the same number of observations from the minority and majority classes.
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=len(df_majority),  # to match majority class
                                 random_state=123)  # reproducible results ??

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
print("Before::", train.label.value_counts())
print("After::", df_upsampled.label.value_counts())

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

#convert text data to numerical data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vect = CountVectorizer()
tf_train=vect.fit_transform(X_train)  #train the vectorizer, build the vocablury -- fit and transform
tf_test=vect.transform(X_test)  #get same encodings on test data as of vocabulary built -- only transform

tf_test_nolabel=vect.transform(test_nolabel.tweet) # -- only transform

#fit the model
model.fit(X=tf_train,y=y_train)

expected = y_test
predicted=model.predict(tf_test)

from sklearn import metrics

# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))

predicted_nolabel=model.predict(tf_test_nolabel)

test_custom=pd.DataFrame(["racist", "white judge trial", "it is a horrible incident", "@user #white #supremacists want everyone to see the new â  #birdsâ #movie â and hereâs why", " @user #white #supremacists want everyone to see the new â  #birdsâ #movie â and hereâs why", "@user  at work: attorneys for white officer who shot #philandocastile remove black judge from presiding over trial. htâ¦"])
tf_custom = vect.transform(test_custom[0])
print(model.predict(tf_custom))


