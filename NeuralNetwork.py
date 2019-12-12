import numpy as np
import random

class NeuralNetwork:
    def __init__(self, layers):
        weight_shapes = [(a, b) for a, b in zip(layers[1:], layers[:-1])]
        self.num_layers = len(layers)
        self.weights = [np.random.standard_normal(s)/(s[1]**0.5) for s in weight_shapes]
        self.biases = [np.zeros((s, 1)) for s in layers[1:]]

    def predict(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.sigmoid(np.matmul(w, x) + b)
        return x

    def train(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_network(batch, learning_rate)
            if test_data is not None:
                correct = self.evaluate(test_data)
                print("Epoch {0}: Accuracy {1}".format(i, correct))
            else:
                print("Epoch {0} finished".format(i))

    def update_network(self, batch, learning_rate):
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in batch:
            delta_delta_w, delta_delta_b = self.backprop(x, y)
            delta_w = [dw + ddw for dw, ddw in zip(delta_w, delta_delta_w)]
            delta_b = [db + ddb for db, ddb in zip(delta_b, delta_delta_b)]
        self.weights = [w-(learning_rate/len(batch) * nw) for w, nw in zip(self.weights, delta_w)]
        self.biases = [b-(learning_rate/len(batch) * nb) for b, nb in zip(self.biases, delta_b)]

    def backprop(self, x, y):
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        zs = []
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            x = np.dot(w, x) + b
            zs.append(x)
            x = self.sigmoid(x)
            activations.append(x)

        delta = (activations[-1] - y) * self.sigmoid_prime(zs[-1])
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        delta_b[-1] = delta

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
            delta_b[-l] = delta

        return delta_w, delta_b

    def evaluate(self, test_data):
        correct = 0
        total = 0
        for x, y in test_data:
            prediction = self.predict(x)
            if np.argmax(prediction) == np.argmax(y):
                correct += 1
            total += 1
        return correct/total

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
