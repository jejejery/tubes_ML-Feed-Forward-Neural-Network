from lib.Layer import Layer
import numpy as np

class OutputLayer(Layer):
    def __init__(self, name, input_shape, output_shape, weights, activation_function):
        super().__init__(name, "output", input_shape, output_shape, weights, activation_function)

    #override
    def forward_propagation(self, input_array : np.array):
        return super().forward_propagation(input_array=input_array)
        #make sure input dimension is 2D
       

    def backward_propagation(self):
        return super().backward_propagation()