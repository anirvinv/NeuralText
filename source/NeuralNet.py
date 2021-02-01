import numpy as np
import random

random.seed(1)

def reLU(num):
    return max(0, num)

class Neuron(object):
    def __init__(self, input):
        self.input = input
        self.weights = np.ones(len(input)) * random.randint(1,20)
        self.bias = random.randint(1,20)

    def output(self):
        a = np.dot(self.input, self.weights) + self.bias
        return reLU(a)

    def __repr__(self):
        return f"{{weights: {self.weights}, bias: {self.bias}}}"


class Layer(object):
    def __init__(self, n_neurons, input):
        self.input = input
        self.neurons = []
        for _ in range(n_neurons):
            neuron = Neuron(input)
            self.neurons.append(neuron)

    def output(self):
        output = []
        for neuron in self.neurons:
            output.append(neuron.output()) 
        return output

    def __repr__(self):
        return str(self.neurons)

class Network(object):
    def __init__(self, features, n_layers, neurons_per_layer):
        self.layers = []
        self.outputs = []
    #Initializes all the dimensions
        self.layers.append(Layer(neurons_per_layer[0], features[0]))
        for i in range(1, n_layers):
            self.layers.append(Layer(neurons_per_layer[i], self.layers[i-1].output()))
        if not len(self.layers[-1].neurons) == 1:
            self.layers.append(Layer(1, self.layers[-1].output()))
        
    def predicted_values(self):
        self.outputs.append(self.layers[-1].output())

        for index, feature in enumerate(features):
            if index == 0:
                continue
            else:
                self.layers[0].input = feature
                print(self.layers[0].output())
                for i in range(1, len(self.layers)):
                    self.layers[i].input = self.layers[i-1].output()
                out = self.layers[-1].output()
                self.outputs.append(out)
        return self.outputs

    


    def __repr__(self): 
        return f"{['-----'+layer.__repr__() + '-----' for layer in self.layers]}"
        

features = [[1,2,3,4], [5,6,7,8]]

net = Network(features, 3, (4,2,3,4))

print(net)
# print(net.predicted_values())