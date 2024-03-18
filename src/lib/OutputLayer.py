from lib.Layer import Layer
from lib.ActivationFunction import ActivationFunction
import numpy as np

class OutputLayer(Layer):
    def __init__(self, name, input_shape, output_shape, weights, activation_function):
        super().__init__(name, "output", input_shape, output_shape, weights, activation_function)

    #override
    def forward_propagation(self, input_array : np.array):
        super().forward_propagation()
        #make sure input dimension is 2D
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
        self.current_output = ActivationFunction.get_activation_function(self.activation_function)(self.pre_activation(input_array))
        return self.current_output
        
        

    def backward_propagation(self):
        return super().backward_propagation()