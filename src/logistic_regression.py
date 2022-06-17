# Logistic regression python file
# Contains all the function to do a logistic regression from scratch

import numpy as np


class LogisticRegression:
    def __init__(self, nb_it, lr, X, Y):  # number opf iteration, learning rate, X, Y
        self.iterations = nb_it
        self.learning_rate = lr
        self.nb_training_exemple, self.nb_feature = X.shape
        self.weight = np.zeros(self.nb_feature)
        self.bias = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.fit()

    def fit(self):
        print(self.X.dot(self.weight))
        print(type(self.bias))
        """
        A = 1/(1+np.exp(-(self.X.dot(self.weight)+self.bias)))

        # Calculation of the gradient
        temp = np.reshape(A-self.Y.T, self.nb_training_exemple)
        dW = np.dot(self.X.T, temp)/self.nb_training_exemple
        db = np.sum(temp)/self.nb_training_exemple

        # New weight
        self.weight = self.weight - self.learning_rate * dW
        self.bias = self.bias - self.learning_rate * db"""

    def predict(self, X):
        Z = 1/(1+np.exp(-(X.dot(self.weight)+self.bias)))
        Y = np.where(Z > 0.5, 1, 0)
        return Y

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


