
class Layer:
    """
    Layer class to store the layer information
    Will inherited by other layer classes (HiddenLayer and OutputLayer)

    Attributes:
    name: Name of the layer
    layer_type: Type of the layer
    input_shape: Shape of the input
    output_shape: Shape of the output
    weights: matrix of weights (input_shape x output_shape)
    biases: array of biases (output_shape, 1)

    ilustration:
    # an example of a layer with 3 inputs and 2 outputs
    =========
    x1--|
        |--> neuron_1 --> output_1 
    x2--|
        |--> neuron_2 --> output_2
    x3--|
    ===========

    ======= EXAMPLE CODE =======
    inputs = np.array([input1, input2, input3])

    # weight matrix: (3 x 2)
    weights_input_output = np.array([[w11, w12],
                                    [w21, w22],
                                    [w31, w32]])

    # the_bias
    bias_output = np.array([b1, b2])

    # do the matrix multiplication + bias
    output_input = np.dot(inputs, weights_input_output) + bias_output

    # finalize with activation function
    final_output = activation_function(output_input)
    ============================


    """
    def __init__(self, name, layer_type, input_shape, output_shape, weights, biases):
        self.name = name
        self.layer_type = layer_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = weights
        self.biases = biases

    """
        Forward propagation of the layer
        for 1st milestone
    """
    def forward_propagation(self):
        print("forward propagation of the layer is not implemented yet")

    
    """
        Backward propagation of the layer
        for 2nd milestone
    """
    def backward_propagation(self):
        print("backward propagation of the layer is not implemented yet")
        
        