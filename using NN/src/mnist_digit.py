import numpy as np

import random as rn

import data_load


def sigmoid(value):
        return 1.0/(1.0+np.exp(-value))

def sigmoid_derivative(value):
        return sigmoid(value)*(1-sigmoid(value))


class Digit_classifier(object):

    def __init__(self,sizes):
        self.number_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[0:])]


    def update_mini_batch(self, mini_batch, learning_rate):
        l = len(mini_batch)
        cost_der_wrt_bias = [np.zeros(b.shape) for b in self.biases]
        cost_der_wrt_weig = [np.zeros(w.shape) for w in self.weights]
        for train_x, train_out in mini_batch:

            delta_b = [np.zeros(b.shape) for b in self.biases]
            delta_w = [np.zeros(w.shape) for w in self.weights]
            activations = [train_x]
            z_values = []
            for i in range(self.number_layers - 1):
                z = np.dot(self.weights[i],activations[i]) + self.biases[i]
                activation = sigmoid(z)
                z_values.append(z)
                activations.append(activation)

            cost_der_wrt_zL = self.cost_derivative(activation,train_out) * \
                 sigmoid_derivative(z)
            delta_b[-1] = cost_der_wrt_zL
            delta_w[-1] = np.dot(cost_der_wrt_zL , activations[-2].transpose())
            cost_der_wrt_z_l = cost_der_wrt_zL
            for layer in range(2,self.number_layers):
                cost_der_wrt_z_l = np.dot(self.weights[1-layer].transpose() , cost_der_wrt_z_l) * sigmoid_derivative(z_values[-layer])
                delta_b[-layer] = cost_der_wrt_z_l
                delta_w[-layer] = np.dot(cost_der_wrt_z_l,activations[-layer-1].transpose())

            cost_der_wrt_bias = [cdb+db for cdb, db in zip(cost_der_wrt_bias, delta_b)]
            cost_der_wrt_weig = [cdw+dw for cdw, dw in zip(cost_der_wrt_weig, delta_w)]
        self.weights = [w-(learning_rate/l)*nw    for w, nw in zip(self.weights, cost_der_wrt_weig)]
        self.biases = [b-(learning_rate/l)*nb     for b, nb in zip(self.biases, cost_der_wrt_bias)]


    def Gradient_descent(self, epochs , mini_batch_size , training_data , learning_rate , test_data):
        num_train = len(training_data)
        num_test = len(test_data)
        for i in range(epochs):
            rn.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]  for k in range(0, num_train , mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), num_test))


    def feedforwading(self,activation):
        for i in range(self.number_layers - 1):
            activation = sigmoid(np.dot(self.weights[i],activation) + self.biases[i])
        return activation


    def evaluate(self, test_data):
        result = [(np.argmax(self.feedforwading(x)), y)   for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in result)


    def cost_derivative(self, output_activations, y):
        return (output_activations-y)



training_data , validation_data , test_data = data_load.load_data()

MNIST_DIGIT_CLASSIFIER = Digit_classifier([784,40,30,10])

MNIST_DIGIT_CLASSIFIER.Gradient_descent(30,10,training_data,0.1,test_data)
