from lib.Layer import Layer

class OutputLayer(Layer):
    def __init__(self, name, input_shape, output_shape, weights, biases):
        super().__init__(name, "output", input_shape, output_shape, weights, biases)

    #override
    def forward_propagation(self):
        return super().forward_propagation()

    def backward_propagation(self):
        return super().backward_propagation()