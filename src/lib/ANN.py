
import numpy as np
from lib.Layer import Layer
from lib.HiddenLayer import HiddenLayer
from lib.OutputLayer import OutputLayer

class ANN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.layers = []
        self.output_size = output_size

    def add(self, layer):
        #if the last layer is output layer, then it's not valid to add another hidden layer
        if(len(self.layers) > 0 and self.layers[-1].layer_type == "output"):
                raise ValueError("OutputLayer already exist!")
        elif isinstance(layer, HiddenLayer):
            self.layers.append(layer)
        elif isinstance(layer, OutputLayer):
            self.layers.append(layer)
        else:
            raise ValueError("Invalid layer type. Only HiddenLayer or OutputLayer allowed.")

    def debug(self):
        print("=====================================")
        for layer in self.layers:
            layer.debug()
            if(layer.layer_type != "output"): print("_____________________________________")
        print("=====================================")
            

    
    """
    Implementation of the forward propagation algorithm

    Arguments:
    X -- Input data (np.array, shape: (input_size, m) 2D array)
    """
    def forward_propagation(self, X : np.array):
        A = X
        for layer in self.layers:
            A = layer.forward_propagation(A)
        return A