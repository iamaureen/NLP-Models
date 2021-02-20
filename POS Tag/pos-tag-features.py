import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re
import matplotlib.pyplot as plt
from collections import Counter

train = pd.read_csv("disaster-train.csv")
test = pd.read_csv("disaster-test.csv")

test["isTrain"] = False

full = pd.concat([train, test])
print('line 14 ::: ', full.isnull().any())
full = full.fillna(method='ffill')

def get_at(row):
    return re.findall("@[\w]+", row["text"])


def get_http(row):
    return re.findall("http[\:\/\.\w]+", row["text"])


def get_hashtags(row):
    return re.findall("#[\w]+", row["text"])


def number_of_tags(row):
    return len(row["tags"])


def number_of_links(row):
    return len(row["links"])


def number_of_hashs(row):
    return len(row["hashtags"])


def clean_text(row):
    clean = row["text"]

    if len(row["tags"]) != 0:
        for word in row["tags"]:
            clean = clean.replace(word, "")

    if len(row["links"]) != 0:
        for word in row["links"]:
            clean = clean.replace(word, "")

    # only remove the # symbol
    clean = clean.replace("#", "").replace("/", "").replace("(", "").replace(")", "")

    return clean.strip()


full["tags"] = full.apply(lambda row: get_at(row), axis=1)
full["links"] = full.apply(lambda row: get_http(row), axis=1)
full["hashtags"] = full.apply(lambda row: get_hashtags(row), axis=1)

full["number_of_tags"] = full.apply(lambda row: number_of_tags(row), axis=1)
full["number_of_links"] = full.apply(lambda row: number_of_links(row), axis=1)
full["number_of_hashs"] = full.apply(lambda row: number_of_hashs(row), axis=1)

full["clean_text"] = full.apply(lambda row: clean_text(row), axis=1)
# print(full.sample(5))

#We have cleaned our texts and stored links, hashtags and tags, it's time for the real deal. We'll first tokenize our texts and use them to get our grammatical classes.
from nltk.tokenize import word_tokenize

def get_tokens(row):
    return word_tokenize(row["clean_text"].lower())

full["tokens"] = full.apply(lambda row: get_tokens(row), axis = 1)
# print(full.sample(5, random_state = 4))

s = ["screams", "in", "the", "distance"]

def get_postags(row):
    postags = nltk.pos_tag(row["tokens"])
    list_classes = list()
    for word in postags:
        list_classes.append(word[1])

    return list_classes


full["postags"] = full.apply(lambda row: get_postags(row), axis=1)
# print(full.sample(5, random_state=4))
# nltk.help.upenn_tagset('NNS')
#we have the POS tags for every text. There are lots of categories and I'll focus only in a few of them: NN, RB, VB, JJ

def find_no_class(count, class_name=""):
    total = 0
    for key in count.keys():
        if key.startswith(class_name):
            total += count[key]

    return total


def get_classes(row, grammatical_class=""):
    count = Counter(row["postags"])
    return find_no_class(count, class_name=grammatical_class) / len(row["postags"])


full["freqAdverbs"] = full.apply(lambda row: get_classes(row, "RB"), axis=1)
full["freqVerbs"] = full.apply(lambda row: get_classes(row, "VB"), axis=1)
full["freqAdjectives"] = full.apply(lambda row: get_classes(row, "JJ"), axis=1)
full["freqNouns"] = full.apply(lambda row: get_classes(row, "NN"), axis=1)

# print(full.sample(5))

full.loc[full["target"] == 0.0, "freqVerbs"].hist(alpha = 0.5);
full.loc[full["target"] == 1.0, "freqVerbs"].hist(alpha = 0.5);

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

x = full.loc[:, ["number_of_tags", "number_of_links", "freqAdverbs", "freqVerbs", "freqAdjectives", "freqNouns"]]
y = full.loc[:, "target"]
#Y = np.asarray(full.loc[:, "target"], dtype=np.float64)

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(x,full['target'])

print('line 129 :: ', y.dtype)
print('line 130 :: ',type(y))
print('line 131 :: ',y)
print('line 132 :: ',type(y[1]))
print('line 133 :: ',y[14:20])


from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# create model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 5205
                                                   )

clf = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=5, random_state=42)
    #     clf = RandomForestClassifier(random_state = 42)

#standard scaler
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print(accuracy_score(y_test, preds))

print(confusion_matrix(y_test, preds))
#
# for train_index, test_index in skf.split(x,y):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_test = x.loc[train_index], x.loc[test_index]
#     y_train, y_test = y.loc[train_index], y.loc[test_index]
#
#
#     clf = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=5, random_state=42)
#     #     clf = RandomForestClassifier(random_state = 42)
#
#     clf.fit(x_train, y_train)
#     preds = clf.predict(x_test)
#
#     print(accuracy_score(y_test, preds))
#
#     print(confusion_matrix(y_test, preds))
#
# total_preds = clf.predict(x)
# print("Confusion Matrix:")
# confusion_matrix(y, total_preds)