#https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73
import re, os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# get data from: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# put into the same folder
def load_train_test_imdb_data(data_dir):
    """Loads the IMDB train/test datasets from a folder path.
    Input:
    data_dir: path to the "aclImdb" folder.

    Returns:
    train/test datasets as pandas dataframes.
    """

    data = {}
    for split in ["train", "test"]:
        data[split] = []
        for sentiment in ["neg", "pos"]:
            score = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, split, sentiment)
            file_names = os.listdir(path)
            for f_name in file_names:
                with open(os.path.join(path, f_name), "r") as f:
                    review = f.read()
                    data[split].append([review, score])


    np.random.shuffle(data["train"])
    data["train"] = pd.DataFrame(data["train"],
                                 columns=['text', 'sentiment'])

    np.random.shuffle(data["test"])
    data["test"] = pd.DataFrame(data["test"],
                                columns=['text', 'sentiment'])

    return data["train"], data["test"]

def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """


    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    # print(text)
    return text

# clean_text("<div>This is not a sentence.<\div>").split()
#
# # small sample example to test, not with real data.
# # training_texts = [
# #     "This is a good cat",
# #     "This is a bad day"
# # ]
# #
# # test_texts = [
# #     "This day is a good day"
# # ]
# #
# # # this vectorizer will skip stop words
# # vectorizer = CountVectorizer(
# #     stop_words="english",
# #     preprocessor=clean_text
# # )
# #
# # # fit the vectorizer on the training text
# # vectorizer.fit(training_texts)
# #
# # # get the vectorizer's vocabulary
# # inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
# # vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
# #
# # # vectorization example
# # pd.DataFrame(
# #     data=vectorizer.transform(test_texts).toarray(),
# #     index=["test sentence"],
# #     columns=vocabulary
# # )
#
# # the following part uses imdb data set to test
# feature vectors that result from BOW are very large (80, 000 dimensional).
# # Let’s train a linear SVM classifier.
#
train_data, test_data = load_train_test_imdb_data(
    data_dir="aclImdb/")

# Transform each text into a vector of word counts - BOW: bag of Words
# CountVectorizer allows to perform vectorization, it also allows us to remove stop words.
# we also pass the custom pre-processing function to automatically clean the text before it
# is vectorized
# vectorizer = CountVectorizer(stop_words="english",
#                              preprocessor=clean_text)

# Transform each text into a vector of word counts **
# improvement: vectorization step.
# issue: some biases attached with only looking at how many times a word occurs in a text.
# In particular, the longer the text, the higher its features (word counts) will be.
# solution: (1) Term Frequency (TF) instead of word counts and divide the number of occurrences by the sequence length
# We can also downscale these frequencies so that words that occur all the time (e.g., topic-related or stop words) have lower values.
# This downscaling factor is called Inverse Document Frequency (IDF) and is equal to the logarithm of the inverse word document frequency.
# solution: (2) further improve the model by providing more context. considering every word independently can lead to some errors.
# For instance, if the word good occurs in a text, we will naturally tend to say that this text is positive, even if the actual expression that occurs is actually
# not good. These mistakes can be easily avoided with the introduction of N-grams.

vectorizer = TfidfVectorizer(stop_words="english",
                             preprocessor=clean_text,
                             ngram_range=(1, 2))

# WHY VECTORIZER.FIT_TRANSFORM
# https://www.quora.com/What-exactly-does-the-fit_transform-function-do-to-your-data-explanatory-variable
training_features = vectorizer.fit_transform(train_data["text"])
test_features = vectorizer.transform(test_data["text"])

# Training
model = LinearSVC()
model.fit(training_features, train_data["sentiment"])
y_pred = model.predict(test_features)

# Evaluation
acc = accuracy_score(test_data["sentiment"], y_pred)

print("Accuracy on the IMDB dataset: {:.2f}".format(acc*100))


