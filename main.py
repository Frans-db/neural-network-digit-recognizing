from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import NeuralNetwork

(train_x, train_y), (test_x, test_y) = mnist.load_data()

nn = NeuralNetwork.NeuralNetwork([28*28, 28, 10])
nn.train(np.array([(x.reshape((28*28,1)),y) for x,y in zip(train_x, train_y)]), 1, 100, 3)

for i in range(0, len(test_x)):
    prediction = nn.predict(np.array(test_x[i].reshape((28*28,1))))
    print(prediction)
    plt.imshow(test_x[i], cmap='Greys')
    plt.show()