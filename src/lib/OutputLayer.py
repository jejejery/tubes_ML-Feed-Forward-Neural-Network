from lib.Layer import Layer
from lib.LossFunction import LossFunction
from lib.ActivationFunction import ActivationFunction
import numpy as np

class OutputLayer(Layer):
    def __init__(self, name, input_shape, output_shape, weights, activation_function, expected_output : np.array):
        super().__init__(name, "output", input_shape, output_shape, weights, activation_function)
        #expected output, np.array 2D
        self.expected_output = expected_output
        self.loss = 0

    #override
    def forward_propagation(self, input_array : np.array):
        return super().forward_propagation(input_array=input_array)
       
        #make sure input dimension is 2D
       
    # update delta weights matrix for 1 mini-batch
    def backward_propagation(self):
        super().backward_propagation() #abstract method of Layer

        #1st, compute the gradient the neuron

        #partial derivate of the loss function with respect to the output of the neuron
        
        '''
            dE/dw = dE/dOut * dOut/dNet * dNet/dw
        '''

        # 1. dE/dOut, np.array 2d, exclude softmax
        de_dNet = None
        if self.activation_function == "relu" or self.activation_function == "sigmoid" or self.activation_function == "linear":
            de_dOut = LossFunction.mean_squared_error_derivative(self.expected_output, self.current_output)
            self.loss = LossFunction.mean_squared_error(self.expected_output, self.current_output)

            # 2. dOut/dNet, np.array 2d, exclude softmax
            dOut_dNet = ActivationFunction.get_activation_derivative(self.activation_function)(self.current_output)
            # 3. de_dNet = de_dOut * dOut_dNet
            de_dNet = de_dOut * dOut_dNet
            # output: [grad_data1, grad_data2, ...]

        # calculate de_dNet  for softmax
        elif self.activation_function == "softmax":
            de_dNet = self.softmax_gradient()
            self.loss = LossFunction.cross_entropy(self.current_output)
              
        # 3. dNet/dw
        # dNet/dw = self.current_input

        # evaluate the batch to update the weights
        # dE/dw = dE/dOut * dOut/dNet * dNet/dw = sum_of_gradients * self.current_input
        # for 0th row of the weights matrix (bias), dNet/dw = 1   
        
        temp = np.insert(self.current_input, 0, np.ones(self.current_input.shape[0]), axis=1)

        # iterate every input in the batch
        for i in range(len(de_dNet)):
            # update the delta weights
            self.delta_weights += np.outer(temp[i], de_dNet[i])
            


        return de_dNet
    

    # compute the gradient of the softmax function
    def softmax_gradient(self):
        #find argmax of each row for the expected output
        pk = np.argmax(self.expected_output, axis=1)
        
        current_output_clone = np.array(self.current_output)
        # sort the current output in descending order
        -np.sort(-current_output_clone, axis=1)
        for i in range(len(current_output_clone)):
            current_output_clone[i][pk[i]] = -(1-current_output_clone[i][pk[i]])

        return current_output_clone