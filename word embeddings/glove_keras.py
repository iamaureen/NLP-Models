
from numpy import array
import numpy as np


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# https://medium.com/analytics-vidhya/text-classification-using-word-embeddings-and-deep-learning-in-python-classifying-tweets-from-6fe644fcfc81
# prepare tokenizer
t = Tokenizer() #constructs unique word dictionary and assigns an integer to every word
t.fit_on_texts(docs)
print(t.word_index)
vocab_size = len(t.word_index) + 1 ## input dimension = total vocab size + 1
# convert the texts into indexed lists. These indexes represent the key values in the tokenizer created dictionary
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)

# The text_to_sequence() method gives us a list of lists where each item has different dimensions
# and is not structured. Any machine learning model needs to know the number of feature dimensions
# and that number must be the same both for training and predictions on new observations.
# To convert the sequences into a well-structured matrix for deep learning training
# we will use the pad_sequances() method from Keras:

# pad documents to a max length of 4 words
max_length = np.max([len(text.split()) for text in docs])
print('max length :: ', max_length)
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post') #padding='post' pads 0 at the end, else pads in the beginning
print(padded_docs)

# load the whole embedding into memory
embeddings_index = dict()
f = open('/Users/isa14/Downloads/glove.twitter.27B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
#print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

## create the LSTM model
model_lstm = Sequential()
model_lstm.add(Embedding(vocab_size, 100, input_length=max_length, weights=[embedding_matrix], trainable=False))
# # model_lstm.add(Dropout(0.2))
# # model_lstm.add(Conv1D(64, 5, activation='relu'))
# # model_lstm.add(MaxPooling1D(pool_size=4))
model_lstm.add(LSTM(100))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit train data
model_lstm.fit(padded_docs, labels, validation_split=0.4, epochs = 3)
#
# # evaluate the model
loss, accuracy = model_lstm.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))