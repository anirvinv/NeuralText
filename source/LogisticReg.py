import numpy as np
import random


def sigmoid(num):
    return 1 / (1 + np.e**(-1 * num))


class Logistic(object):
    def __init__(self, lr=0.01, n_iters=1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.weights = np.ones(len(X[0])) * random.random()
        self.bias = random.randint(0, 10)

        for _ in range(self.n_iters):
            total = 0
            gradient = np.array([])
            dw = 0
            db = 0

            for d in X:
                total += np.dot(self.weights, d)
            total = sigmoid(total)

            for i in range(len(self.X[0])):
                for index, sample in enumerate(y):
                    dw += (total - sample) * self.X[index][i]
                    db += total - sample
                gradient = np.concatenate((gradient, np.array([dw])), axis=0)

            self.weights -= gradient * self.lr
            self.bias -= db * self.lr

    def predict(self, X):
        prediction = np.array([])

        for x in X:
            sol = sigmoid(np.dot(x, self.weights) + self.bias)
            prediction = np.concatenate((prediction, np.array([sol])))

        return prediction


X = [[1, 2], [3, 4]]
y = [0, 1]

model = Logistic()


model.fit(X, y)

print(model.predict([[1, 2], [10, 12]]))
