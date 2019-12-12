from keras.datasets import mnist
import random
import matplotlib.pyplot as plt
import numpy as np
import NeuralNetwork

# get data from keras
print('-=-Getting training data-=-')
(train_x, train_y), (test_x, test_y) = mnist.load_data()

training_data = []
# setup training data
for i in range(0, len(train_x)):
    y = np.zeros((10,1))
    y[train_y[i]-1]=1
    training_data.append((train_x[i].reshape((28*28,1))/255,y))

random.shuffle(training_data)
test_data = training_data[:5000]
training_data = training_data[5000:]

# create network
nn = NeuralNetwork.NeuralNetwork([28*28, 28, 10])
# train network
print('-=-Training network-=-')
nn.train(training_data, 5, 40, 3, test_data=test_data)

# test network
print('-=-Testing network. Press q to go to next digit, terminate program to stop-=-')
for i in range(0, len(test_x)):
    prediction = nn.predict(np.array(test_x[i].reshape((28*28,1)))/255)
    print('Prediction: {0}. Actual digit: {1}'.format(np.argmax(prediction)+1, test_y[i]))
    plt.imshow(test_x[i], cmap='Greys')
    plt.show()