from lib.Layer import *

class HiddenLayer(Layer):
    def __init__(self, name, input_shape, output_shape, weights, activation_function):
        super().__init__(name, "hidden", input_shape, output_shape, weights, activation_function)

    #override
    #override
    def forward_propagation(self, input_array : np.array):
        self.current_output = super().forward_propagation(input_array=input_array)
        return self.current_output
        #make sure input dimension is 2D

    # gradient is 2D array, each row is the gradient of each data in the batch
    # weights is the weights of the next layer
    # return (de_dNet, weights)
    def backward_propagation(self, gradient, weights):
        super().backward_propagation()

        # 1. dE/dOut, np.array 2d

        de_dOut = gradient @ weights.T


        # 2. dOut/dNet, np.array 2d
        dOut_dNet = ActivationFunction.get_activation_derivative(self.activation_function)(self.current_output)

        # 3. de_dNet = de_dOut * dOut_dNet
        de_dNet = de_dOut * dOut_dNet
        # output: [grad_data1, grad_data2, ...]

        # 4. dNet/dw
        # dNet/dw = self.current_input
        temp = np.insert(self.current_input, 0, np.ones(self.current_input.shape[0]), axis=1)

        # evaluate the batch to update the weights
        # dE/dw = dE/dOut * dOut/dNet * dNet/dw = sum_of_gradients * self.current_input
        # for 0th row of the weights matrix (bias), dNet/dw = 1
        for i in range(len(de_dNet)):
            # update the delta weights
            self.delta_weights += np.outer(temp[i], de_dNet[i])
    
        

        return de_dNet, np.array(self.weights[1:], dtype=np.float64)

    def predict(self, input_array : np.array):
        result = super().predict(input_array=input_array)
        return result
        
        

        

