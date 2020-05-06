import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

filename = "uci-news-aggregator.csv"

#read data file
data = pd.read_csv('uci-news-aggregator.csv', usecols=['TITLE', 'CATEGORY'])

#M class has way less data than the orthers, thus the classes are unbalanced.
category_counts = data.CATEGORY.value_counts()
print(category_counts)

#balance the datasets
num_of_categories = 45000 #why 45,000? the lowest count of the type is 45639, so to balance it maybe used
shuffled = data.reindex(np.random.permutation(data.index))
# print(shuffled) #shuffles the data -- original data seems sorted
# the following gets equal number of rows for each type
# steps: check for each category, get the rows that only has particular category true, then get the firsst 45,000 rows
e = shuffled[shuffled['CATEGORY'] == 'e'][:num_of_categories]
b = shuffled[shuffled['CATEGORY'] == 'b'][:num_of_categories]
t = shuffled[shuffled['CATEGORY'] == 't'][:num_of_categories]
m = shuffled[shuffled['CATEGORY'] == 'm'][:num_of_categories]
# concat all the rows together
concated = pd.concat([e,b,t,m], ignore_index=True)
# print(concated.shape)
#
# #Shuffle the dataset #why shuffle again?
concated = concated.reindex(np.random.permutation(concated.index))
# create new column with label
concated['LABEL'] = 0

# #One-hot encode the lab
concated.loc[concated['CATEGORY'] == 'e', 'LABEL'] = 0
concated.loc[concated['CATEGORY'] == 'b', 'LABEL'] = 1
concated.loc[concated['CATEGORY'] == 't', 'LABEL'] = 2
concated.loc[concated['CATEGORY'] == 'm', 'LABEL'] = 3
# print(concated['LABEL'][:10])

labels = to_categorical(concated['LABEL'], num_classes=4) #one hot encoding
# print(labels[:10])

# #Use keys() function to find the columns of a dataframe.
print(concated.shape)
if 'CATEGORY' in concated.keys():
    concated.drop(['CATEGORY'], axis=1) #but why drop?

print(concated.shape) #shape is the same for concated, so what dropped?

epochs = 1
emb_dim = 128
batch_size = 256

n_most_common_words = 8000
max_len = 130
tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# #https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do
tokenizer.fit_on_texts(concated['TITLE'].values)
sequences = tokenizer.texts_to_sequences(concated['TITLE'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)

model = Sequential()
#model.add() keeps adding layers in the model
model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.7))
model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])

accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

#testing. using a different sentence for texts_to_sequences (follow the stackoverflow link above)
txt = ["Regular fast food eating linked to fertility issues in women"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_len)
pred = model.predict(padded)
labels = ['entertainment', 'bussiness', 'science/tech', 'health']
print(pred, labels[np.argmax(pred)])