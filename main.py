from http.client import responses
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
import pickle

with open("intents.json") as file:
    data = json.load(file)

# print(data)

# empty lists
try:
    with open("date.pickle", "rb") as f:
        words,labels,training,output = pickle.load(f)

except:
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

    with open("data.pickle", "wb") as f:
        pickle.dump((words,labels,training, output),f)



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

try:
    model.load("model.tflearn")

except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

                
    return numpy.array(bag)


def chat():
    print("start talking with the bot! (type quite to stop)")

    while True:
        inp = input("You: ")
        if inp.lower == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        # print(tag)

        if results[results_index] > .7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))

        else:
            print("I didnt get that try again")


chat()