
import numpy as np
from lib.Layer import Layer
from lib.HiddenLayer import HiddenLayer
from lib.OutputLayer import OutputLayer

class ANN:
    def __init__(self, input_size, output_size, learning_rate=1e-2, max_epoch=1000, tolerance=1e-5):
        self.input_size = input_size
        self.layers = []
        self.output_size = output_size
        self.learning_rate = learning_rate

    def add(self, layer):
        #check if the input shape of the new layer is the same as the output shape of the last layer (if filled)
        if(len(self.layers) > 0 ):
            if(layer.input_shape != self.layers[-1].output_shape):
                raise ValueError(f"Invalid input shape. Expected: {self.layers[-1].output_shape}")
        else:
            if(layer.input_shape != self.input_size):
                raise ValueError(f"Invalid input shape. Expected: {self.input_size}")
    
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
        # print(f"loss: {loss(A)}")
        # print turunan dari loss(A)
        return A
    
    def backward_propagation(self):
        #implement the backward propagation
        #for 2nd milestone
        gradient = None
        for layer in reversed(self.layers):
            if gradient is None:
                #for output layer, the gradient is the gradient of the loss function
                gradient = layer.backward_propagation()
            else:
                #for hidden layer, the gradient is the gradient of the next layer
                # gradient = layer.backward_propagation(gradient) =====> to be implemented
                layer.backward_propagation()
            self.update_weights()
        self.reset_delta_weights()

    def update_weights(self):
        for layer in self.layers:
            layer.weights = layer.weights - self.learning_rate * layer.delta_weights
    
    def reset_delta_weights(self):
        for layer in self.layers:
            layer.delta_weights = np.zeros(layer.weights.shape)
        
          
            