#reference: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

# use word embeddings for deep learning in python with keras
# Keras offers an Embedding layer that can be used for neural networks on text data.

# small problem where we have 10 text documents,
# each with a comment about a piece of work a student submitted.
# Each text document is classified as positive “1” or negative “0”.
# This is a simple sentiment analysis problem.
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

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

# we can integer encode each document. This means that as input the Embedding layer will have sequences of integers.
# We could experiment with other more sophisticated bag of word model encoding like counts or TF-IDF.

# Keras provides the one_hot() function that creates a hash of each word as an efficient integer encoding.
# We will estimate the vocabulary size of 50, which is much larger than needed to reduce the probability of collisions from the hash function


# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print("integer encoded documents:")
print(encoded_docs)

#The sequences have different lengths and Keras prefers inputs to be vectorized and all inputs to have the same length.
# We will pad all input sequences to have the length of 4. Again, we can do this with a built in Keras function, in this case the pad_sequences() function.

# pad documents to a max length of 4 words (use max length of the document)
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print("padded versions of each document are printed, making them all uniform length:")
print(padded_docs)

# The Embedding layer is defined as the first hidden layer of a network. It must specify 3 arguments: input_dim, output_dim, and input_length;
# The Embedding has a vocabulary of 50 and an input length of 4. We will choose a small embedding space of 8 dimensions.
# output: The model is a simple binary classification model. Importantly, the output from the Embedding layer will be 4 vectors of 8 dimensions each,
# one for each word. We flatten this to a one 32-element vector to pass on to the Dense output layer.

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

#the accuracy of the trained model is printed, showing that it learned the training dataset perfectly (which is not surprising).

