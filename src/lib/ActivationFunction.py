import numpy as np

"""
Implement as static method
example: output = ActivationFunction("sigmoid", X) 
where X = W.T @ X + b
"""
class ActivationFunction:
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_derivative(Z):
        return Z * (1 - Z)

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        return np.where(Z <= 0, 0, 1)

    @staticmethod
    def tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def tanh_derivative(Z):
        return 1 - np.power(ActivationFunction.tanh(Z), 2)

    @staticmethod
    def softmax(Z):
        if(len(Z.shape) == 1):
            Z = Z.reshape(1, -1)
        expZ = np.exp(Z - np.max(Z))
        # check each expZ element, if 0, then it will be 10^-10
        expZ[expZ == 0] = 1e-10
        return expZ / expZ.sum(axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(Z):
        return ActivationFunction.softmax(Z) * (1 - ActivationFunction.softmax(Z))

    @staticmethod
    def linear(Z):
        return Z

    @staticmethod
    def linear_derivative(Z):
        return np.ones(Z.shape)

    @staticmethod
    def get_activation_function(name):
        if name == "sigmoid":
            return ActivationFunction.sigmoid
        elif name == "relu":
            return ActivationFunction.relu
        elif name == "tanh":
            return ActivationFunction.tanh
        elif name == "softmax":
            return ActivationFunction.softmax
        elif name == "linear":
            return ActivationFunction.linear
        else:
            raise ValueError(f"Invalid activation function: {name}")

    @staticmethod
    def get_activation_derivative(name):
        if name == "sigmoid":
            return ActivationFunction.sigmoid_derivative
        elif name == "relu":
            return ActivationFunction.relu_derivative
        elif name == "tanh":
            return ActivationFunction.tanh_derivative
        elif name == "softmax":
            return ActivationFunction.softmax_derivative
        elif name == "linear":
            return ActivationFunction.linear_derivative
        else:
            raise ValueError(f"Invalid activation function: {name}")
        
    """
    example usage  of 2 datasets that enters layer with 3 neurons:
    sigma = np.array([[-1, 2, 3], [4, 5, 6]])
    output = ActivationFunction.activate("relu", sigma)
    print(output)

    #expected output:
    # [[0 2 3], [4 5 6]]
    """
    @staticmethod
    def activate(name, Z):
        return ActivationFunction.get_activation_function(name)(Z)

    @staticmethod
    def derivative(name, Z):
        return ActivationFunction.get_activation_derivative(name)(Z)



