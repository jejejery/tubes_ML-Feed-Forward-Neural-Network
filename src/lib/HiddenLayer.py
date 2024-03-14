from lib.Layer import *

class HiddenLayer(Layer):
    def __init__(self, name, input_shape, output_shape, weights, biases):
        super().__init__(name, "hidden", input_shape, output_shape, weights, biases)

    #override
    def forward_propagation(self):
        return super().forward_propagation()