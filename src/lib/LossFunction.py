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


    def cross_entropy(target : np.array, output : np.array):
        loss = -(target * np.log(output))
        return np.sum(loss)/len(output)
    

    def mean_squared_error_derivative(target : np.array,output : np.array):
        '''
            dE/dOut = Out - Target

            for 1 row:
        '''
        return output - target

    