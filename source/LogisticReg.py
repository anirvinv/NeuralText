import numpy as np
import random

def sigmoid(num):
    return 1/(1 + np.e**(-1 * num))

class Logistic(object):
    def __init__(self):
        self.weights = []
        self.bias = []
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.weights = np.ones(X[0].size) * random.randint(0, 9)
        self.bias = random.randint(0, 10)
        
        total = 0
        for d in X:
            total += np.dot(d, weights)
        total = sigmoid(total)

    
    def predict(self):
        pass