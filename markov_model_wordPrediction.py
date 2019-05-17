# reference: https://github.com/ashwinmj/word-prediction/blob/master/MarkovModel.ipynb
# Explanation: https://medium.com/ymedialabs-innovation/next-word-prediction-using-markov-model-570fc0475f96
# use of Markov model for word prediction. Specifically 2nd order Markov model is deployed here for next word prediction.
import string
import numpy as np

# Path of the text file containing the training data
training_data_file = 'eminem_songs_lyrics.txt'

#################################
# training start
# helper functions for training
#################################

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('','', string.punctuation))

def add2dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)

def list2probabilitydict(given_list):
    probability_dict = {}
    given_list_length = len(given_list)
    for item in given_list:
        probability_dict[item] = probability_dict.get(item, 0) + 1
    for key, value in probability_dict.items():
        probability_dict[key] = value / given_list_length
    return probability_dict

initial_word = {}
second_word = {}
transitions = {}


# Trains a Markov model based on the data in training_data_file
def train_markov_model():
    for line in open(training_data_file):
        tokens = remove_punctuation(line.rstrip().lower()).split()
        tokens_length = len(tokens)
        for i in range(tokens_length):
            token = tokens[i]

            if i == 0: #this is the first word so no previous word
                #When get() is called, Python checks if the specified key exists in the dict.
                # If it does, then get() returns the value of that key.
                # If the key does not exist, then get() returns the value specified in the second argument to get().
                initial_word[token] = initial_word.get(token, 0) + 1
            else:
                prev_token = tokens[i - 1]
                if i == tokens_length - 1: #this is the last word
                    #in this case, the key in the dictionary is a part of words
                    add2dict(transitions, (prev_token, token), 'END')
                if i == 1: #this stores the second word followed by the first word for each line
                    add2dict(second_word, prev_token, token)
                else:
                    prev_prev_token = tokens[i - 2]
                    # in this case, the key in the dictionary is a part of words
                    add2dict(transitions, (prev_prev_token, prev_token), token)

    # print the values to see how the data structure looks, it will be easier to understand
    # print(second_word)
    # print(transitions)
    # Normalize the distributions
    initial_word_total = sum(initial_word.values())
    for key, value in initial_word.items():
        initial_word[key] = value / initial_word_total

    for prev_word, next_word_list in second_word.items():
        second_word[prev_word] = list2probabilitydict(next_word_list)

    for word_pair, next_word_list in transitions.items():
        transitions[word_pair] = list2probabilitydict(next_word_list)

    print('Training successful.')

train_markov_model()

#################################
# # testing start
# # helper function for testing
#################################

def sample_word(dictionary):
    p0 = np.random.random()
    cumulative = 0
    for key, value in dictionary.items():
        cumulative += value
        if p0 < cumulative:
            return key
    assert(False)

number_of_sentences = 10

# Function to generate sample text
def generate():
    for i in range(number_of_sentences):
        sentence = []
        # Initial word
        word0 = sample_word(initial_word)
        sentence.append(word0)
        # Second word
        word1 = sample_word(second_word[word0])
        sentence.append(word1)
        # Subsequent words until END
        while True:
            word2 = sample_word(transitions[(word0, word1)])
            if word2 == 'END':
                break
            sentence.append(word2)
            word0 = word1
            word1 = word2
        print(' '.join(sentence))

generate()