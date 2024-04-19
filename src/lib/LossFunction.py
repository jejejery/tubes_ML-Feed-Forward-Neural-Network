import numpy as np

class LossFunction:
    @staticmethod
    def mean_squared_error(target : np.array,output : np.array):
        '''
            E = 1/2 * sigma((target - output)^2)

            target : np.array , 2 dimensional
            output : np.array , 2 dimensional

        '''
        temp_sum = 0
        for i in range(len(target)):
            temp_sum += np.sum((target[i] - output[i])**2)
        return 0.5 * temp_sum


    def cross_entropy(output : np.array):

        '''
            Output, np.array 2D
            Output example: [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]

            pk = max(Output)

        '''

        temp_sum = 0
        for i in range(len(output)):
            pk = np.argmax(output[i])
            temp_sum += -np.log(output[i][pk] + 1e-10)
        return temp_sum
    

    def mean_squared_error_derivative(target : np.array,output : np.array):
        '''
            dE/dOut = Out - Target

            for 1 row:
        '''
        return output - target

    @staticmethod
    def get_loss_function(name, target, output):
        if name == "relu" or name == "sigmoid" or name == "linear":
            return LossFunction.mean_squared_error(target, output)
        elif name == "softmax":
            return LossFunction.cross_entropy(output)
        # elif name == "tanh":
        #     not implemented yet
        else:
            raise ValueError("Invalid activation function name")
        
    