import random
import pickle
import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

# get data from keras
print('-=-Getting training data-=-')
# (train_x, train_y), (test_x, test_y) = mnist.load_data()
with np.load('mnist.npz') as data:
    train_x = data['training_images']
    train_y = data['training_labels']

training_data = []
# setup training data
for i in range(0, len(train_x)):
    #y = np.zeros((10,1))
    #y[train_y[i]-1]=1
    training_data.append((train_x[i],train_y[i]))

random.shuffle(training_data)
test_data = training_data[:5000]
training_data = training_data[5000:]

# create network
inputs = 28*28
hidden = [int(inputs**0.5) for i in range(0, 5)]
outputs = 10
nn = NeuralNetwork.NeuralNetwork([inputs] + hidden + [outputs])
# train network
print('-=-Training network-=-')
nn.train(training_data, 25, 40, 3, test_data=test_data)

# test network
print('-=-Testing network. Press q to go to next digit, terminate program to stop-=-')
for i in range(0, len(test_data)):
    prediction = nn.predict(np.array(test_data[i][0]))
    num_predict = np.argmax(prediction)
    num_actual = np.argmax(test_data[i][1])
    print('Prediction: {0}. Actual digit: {1}'.format(num_predict, num_actual))
    if num_predict != num_actual:
        plt.imshow(test_data[i][0].reshape((28,28)), cmap='Greys')
        plt.show()