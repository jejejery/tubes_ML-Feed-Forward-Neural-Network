import numpy as np
from lib.ActivationFunction import ActivationFunction

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
    # input vector: (2 x 4) -> included bias  
    x = np.array([[1, x11, x12, x13], [1, x21, x22, x23]])

    # weight matrix: (4 x 3) -> first row is bias
    # output shape: number of columns of the weight matrix or W.shape[1]
    W = np.array([[wb1, wb2, wb3],
                    [w11, w21, w31],
                    [w12, w22, w32],
                    [w13, w23, w33]])
    

    # do the matrix multiplication
    output_input = x @ W

    # expected output:
    [[wb1 + w11*x11 + w12*x12 + w13*x13, wb2 + w21*x11 + w22*x12 + w23*x13, wb3 + w31*x11 + w32*x12 + w33*x13], -> 1st data
    [wb1 + w11*x21 + w12*x22 + w13*x23, wb2 + w21*x21 + w22*x22 + w23*x23, wb3 + w31*x21 + w32*x22 + w33*x23] -> 2nd data
    ]

    # finalize with activation function, can be relu or something else
    final_output = activation_function(output_input)
    ============================


    """
    def __init__(self, name : str, layer_type : str, input_shape : int, output_shape: int, weights : np.array, activation_function :str):
        self.name = name
        self.layer_type = layer_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        if(weights.shape != (input_shape+1, output_shape)):
            raise ValueError(f"Invalid weight shape: {weights.shape}. Expected: {(input_shape+1, output_shape)}")
        self.weights = weights
        self.delta_weights = np.zeros(self.weights.shape)
        self.activation_function = activation_function
        # initialize when forward propagation is called
        self.current_input = None
        self.current_output = None

    

    """
        Forward propagation of the layer
    """
    def forward_propagation(self, input_array : np.array):
         #make sure input dimension is 2D
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
        self.current_input = input_array
        self.current_output = ActivationFunction.get_activation_function(self.activation_function)(self.pre_activation(input_array))
        return self.current_output
    
    """
    linear combination of the input and the weights
    expected input: (number_of_data, input_shape+1)
    expected weights: (input_shape+1, output_shape)
    expected output: (number_of_data, output_shape)

    """
    def pre_activation(self, input_array : np.array):
        #preprocess the input 
        add_bias = np.insert(input_array, 0, np.ones(input_array.shape[0]), axis=1)
        #do the matrix multiplication
        return add_bias @ self.weights
    
    """
        Backward propagation of the layer
        for 2nd milestone
    """
    def backward_propagation(self):
        return

    def debug(self):
        print(f"Layer: {self.name} | Type: {self.layer_type}", end=" ")
        print(f"| Output shape: {self.output_shape}")
        print(f"Weights:\n {self.weights}")
        
        