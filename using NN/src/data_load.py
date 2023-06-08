import numpy as np

import gzip

import pickle


def vectorize_function(y):

    arr = np.zeros((10,1))

    arr[y] = 1.0

    return arr


def load_data():

    f = gzip.open('../data/mnist.pkl.gz' , 'rb')

    u = pickle._Unpickler( f )

    u.encoding = 'latin1'

    training_data , validation_data , ta = u.load()

    f.close()

    data_x_nice = [np.reshape(x , (784,1)) for x in training_data[0]]

    data_y_nice = [vectorize_function(y) for y in training_data[1]]

    training_data = list(zip(data_x_nice , data_y_nice))

    data_x_nice = [np.reshape(x , (784,1)) for x in validation_data[0]]

    data_y_nice = [vectorize_function(y) for y in validation_data[1]]

    validation_data = list(zip(data_x_nice , data_y_nice))

    data_x_nice = [np.reshape(x , (784,1)) for x in ta[0]]

    test_data = list(zip(data_x_nice , ta[1]))

    return (training_data, validation_data, test_data)
