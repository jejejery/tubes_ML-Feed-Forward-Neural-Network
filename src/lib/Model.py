from lib.ANN import ANN
import numpy as np

class Model:

    def __init__(self, name, ann : ANN):
        self.name = name
        self.ann = ann
        self.test_input = None

    """
    build the model like example testcase in .json file
    """
    def build(self):
        print("building the model")

    def add(self, layer):
        self.ann.add(layer)

    def summary(self):
        print(f"Summary for Model: {self.name}")
        self.ann.debug()

    def predict(self, X : np.array):
        #resize is X is not 2D array
        if(len(X.shape) == 1):
            X = X.reshape(X.shape[0], 1)
        #if input shape invalid
        if(X.shape[1] != self.ann.input_size):
            raise ValueError(f"Invalid input size. Expected: {self.ann.input_size}, got: {X.shape[1]}")
        #return
        return self.ann.forward_propagation(X)
        