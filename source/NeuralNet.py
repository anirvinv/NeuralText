import numpy as np
import random

random.seed(1)

def reLU(num):
    return max(0, num)

class Neuron(object):
    def __init__(self, input):
        self.input = input
        self.weights = np.ones(len(input)) * random.randint(1,20)
        self.bias = random.randint(1,9)

    def change_input(self, input):
        self.input = input

    def output(self):
        a = np.dot(self.input, self.weights) + self.bias
        return reLU(a)

    def __repr__(self):
        return f"{{weights: {self.weights}, bias: {self.bias}}}"

    def visual_representation(self):
        return f"( {self.bias} )"


class Layer(object):
    def __init__(self, n_neurons, input):
        self.input = input
        self.neurons = []
        for _ in range(n_neurons):
            neuron = Neuron(input)
            self.neurons.append(neuron)

    def change_input(self, input):
        self.input = input
        for neuron in self.neurons:
            neuron.change_input(input)

    def output(self):
        output = []
        for neuron in self.neurons:
            output.append(neuron.output()) 
        return output

    def __repr__(self):
        return "--------" + str(self.neurons) + "--------"
    
    def visual_representation(self):
        visual = ""
        for neuron in self.neurons:
            visual += neuron.visual_representation()
        return visual

class Network(object):
    def __init__(self, features, labels, n_layers, neurons_per_layer):
        assert(n_layers == len(neurons_per_layer))
        self.features = features
        self.labels = labels
        self.layers = []
        self.outputs = []
        self.neurons_per_layer = neurons_per_layer
        self.layers.append(Layer(neurons_per_layer[0], features[0]))
        for i in range(1, n_layers):
            self.layers.append(Layer(neurons_per_layer[i], self.layers[i-1].output()))
        if not len(self.layers[-1].neurons) == 1:
            self.layers.append(Layer(1, self.layers[-1].output()))
    def __repr__(self): 
        rep = ""
        for layer in self.layers:
            rep += str(layer.__repr__()) + "\n"
        return rep
    
    def output(self):
        for feature in self.features:
            self.layers[0].change_input(feature)
            for i in range(1, len(self.layers)):
                self.layers[i].change_input(self.layers[i-1].output())
            out = self.layers[-1].output()
            self.outputs.append(out)
        return self.outputs

    def fit(self):
        pass

    def predict(self, X):
        pass

    def visual_representation(self):
        visual = ""
        for index, num in enumerate(self.neurons_per_layer):
            if index == len(self.neurons_per_layer) - 1:
                visual += self.layers[index].visual_representation()
            else:
                visual += self.layers[index].visual_representation() + "\n" + "  |  " * self.neurons_per_layer[index] + "\n"
                visual += "__-__" * self.neurons_per_layer[index ] + "\n"
                visual += "  |  " * self.neurons_per_layer[index + 1] + "\n"
        return "TOP DOWN VISUAL:\n" + visual
        

features = np.array([[1,2,3,4], [0,0,0,0]])
labels = np.array([1, 2, 3])

net = Network(features, labels, 7, (1, 2, 3, 4, 3, 2,1))
print(net.output())
print(net.visual_representation())
