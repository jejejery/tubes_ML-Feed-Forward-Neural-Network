from lib.Layer import *

class HiddenLayer(Layer):
    def __init__(self, name, input_shape, output_shape, weights, activation_function):
        super().__init__(name, "hidden", input_shape, output_shape, weights, activation_function)

    #override
    #override
    def forward_propagation(self, input_array : np.array):
        return super().forward_propagation(input_array=input_array)
        #make sure input dimension is 2D