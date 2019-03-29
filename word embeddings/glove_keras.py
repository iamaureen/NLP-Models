
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
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
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1 ## input dimension = total vocab size + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
f = open('/Users/iamaureen/Documents/glove.6B/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

# define model
# model = Sequential()
# e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
# model.add(e)
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# # compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# # summarize the model
# print(model.summary())
# # fit the model
# model.fit(padded_docs, labels, epochs=50, verbose=0)

# evaluate the model
# loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# print('Accuracy: %f' % (accuracy*100))

## create the LSTM model
model_lstm = Sequential()
model_lstm.add(Embedding(vocab_size, 100, input_length=4, weights=[embedding_matrix], trainable=False))
# model_lstm.add(Dropout(0.2))
# model_lstm.add(Conv1D(64, 5, activation='relu'))
# model_lstm.add(MaxPooling1D(pool_size=4))
model_lstm.add(LSTM(100))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit train data
model_lstm.fit(padded_docs, labels, validation_split=0.4, epochs = 3)

# evaluate the model
loss, accuracy = model_lstm.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))