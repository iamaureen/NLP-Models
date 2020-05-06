# https://www.kaggle.com/kredy10/simple-lstm-for-text-classification/notebook
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping


#Load the data into Pandas dataframe¶
df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')
print(df.head())

#Drop the columns that are not required for the neural network.
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
df.info()

sns.countplot(df.v1)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')
plt.show()

# Create input and output vectors.
# Process the labels.
X = df.v2
Y = df.v1
le = LabelEncoder() #Encode labels with value between 0 and n_classes-1. #define the encoder
Y = le.fit_transform(Y) #use the encoder to encode different classes # it is a row of items
print('fit transdorm :: ', Y.shape)
Y = Y.reshape(-1,1) #https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
# print('after the reshape :: ', Y.shape)

#Split into training and test data.
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

# Process the data
# Tokenize the data and convert the text to sequences.
# Add padding to ensure that all the sequences have the same shape.
# There are many ways of taking the max_len and here an arbitrary length of 150 is chosen.

max_words = 1000
max_len = 150
#num_words: the maximum number of words to keep, based on word frequency. -- https://keras.io/preprocessing/text/
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
# integer encode the documents
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

#Define the RNN structure
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

#Call the function and compile the model.
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

#Fit on the training data.
model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])


#Process the test set data.
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

#Evaluate the model on the test set.
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))