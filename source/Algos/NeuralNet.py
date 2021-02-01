import numpy as np


class Layer_Dense:
    def __init__(self, n_neurons, n_features):
        self.weights = np.random.random((n_neurons, n_features))
        self.bias = np.random.random((n_neurons, 1))

    def forward(self, X):
        self.output = (np.dot(self.weights, X.T)).T

class Activation:
    def soft_forward(self, X):
        exp_vals = np.exp(X)
        total = np.sum(exp_vals, axis=1)
        self.output = exp_vals/total


    def reLU_forward(self, X):
        self.output = max(0, X)


    def sigm_forward(self, X):
        self.output = 1/(1 + np.exp(-X))

X_sample = np.array([[1,2,3],[2,3,4],[3,4,5]])
y_sample = np.array([[1, 3, 2, 1]]).T

layer1  = Layer_Dense(2, 3)
layer2 = Layer_Dense(4, 2)
layer3 = Layer_Dense(3,4)

sigmoid = Activation()
sigmoid2 = Activation()
softmax = Activation()

layer1.forward(X_sample)
sigmoid.sigm_forward(layer1.output)


layer2.forward(sigmoid.output)
sigmoid2.sigm_forward(layer2.output)

layer3.forward(sigmoid2.output)
softmax.soft_forward(layer3.output)


print(np.sum(softmax.output, axis=1))