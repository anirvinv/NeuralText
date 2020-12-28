import numpy as np
import random


class LinearModel(object):
	"""
	Linear regression algorithm made from scratch
	"""
    def __init__(self, x, y, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.X = x
        self.Y = y
        self.coeffs = np.ones(x[0].size) * random.randint(0, 10)
        self.bias = np.random.randint(0, 10)

    def gradient(self):
        """
        Calculates the gradient
        :return:gradient as a numpy array
        """
        labels = self.Y
        features = self.X

        gradient = np.array([])

        dw = 0
        db = 0
        for f in range(len(features[0])):
            for i in range(len(labels)):
                dw += (labels[i][0] - (np.dot(self.coeffs, features[i]) + self.bias)) * -1 * features[i][f]
                db += (labels[i][0] - (np.dot(self.coeffs, features[i]) + self.bias)) * -1
            # dw /= len(features) ~ doesnt matter
            gradient = np.concatenate((gradient, np.array([dw])), axis=0)

        return (gradient, db)

    def step_size(self):
        """
        Calculates and returns the step size
        :return: step_size
        """

        step_size = self.learning_rate * self.gradient()[0]
        bias_step = self.learning_rate * self.gradient()[1]
        return (step_size, bias_step)

    def fit(self):
        """
        Find values of the coefficients that minimize the loss function, LSE
        :return: void
        """

        for _ in range(1000):
            self.coeffs -= self.step_size()[0]
            self.bias -= self.step_size()[1]

        print(self.coeffs, self.bias)

    def predict(self, x):
        """
        Predicts using linear regression algorithm
        :param x: Features
        :return: Predicted label
        """
        return self.bias + np.dot(self.coeffs, x)


x = np.array([[1, 2], [4, 5]])
y = np.array([[6], [18]])
# data = np.concatenate((x, y), axis=1)
#
# print(data)


model = LinearModel(x, y)
model.fit()
print(model.predict(np.array([1, 2])))
