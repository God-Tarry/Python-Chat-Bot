import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
""" from tensorflow.python.framework import ops
ops.reset_default_graph() """
import random
import json

with open("intents.json") as file:
    data = json.load(file)

# print(data)

# empty lists

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])


    if intent["tag"] not in labels:
        labels.append(intent["tag"])



words = [stemmer.stem(w.lower()) for w in words if w not in "?"]

# turns words list into a set - takes all the words make sure there are no duplicates

# list converts the words back into a list

words = sorted(list(set(words)))


# sorts the labels aswell

labels =sorted(labels)



# neural networks only understand numbers

""" 
    to train our model we feed a list - 
    each position in the list will represent weather a word exists or not 
"""

# variables /lists for the training and output

training = []
output = []

out_empty = [0 for _ in range(len(labels))]


for x, doc in enumerate(docs_x):
    bag =[]

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)


""" tf-learn start coding the model """

# tensorflow.reset_default_graph()
tensorflow.compat.v1.reset_default_graph()

# start with input data - length of our framework


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0],), activation="softmax")
net  = tflearn.regression(net)


model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")