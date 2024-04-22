from lib.ANN import ANN
import numpy as np
import pickle as p

class Model:

    def __init__(self, name, ann : ANN):
        self.name = name
        self.ann = ann
        self.train_input = None
        self.train_output = None
        self.valid_input = None
        self.valid_output = None
        self.loses = []


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

    def save(self,filename):
        with open(filename, 'wb') as file:
            p.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            temp = p.load(file)
        return temp

    def predict(self, X : np.array):
        #resize is X is not 2D array
        if(len(X.shape) == 1):
            X = X.reshape(X.shape[0], 1)
        #if input shape invalid
        if(X.shape[1] != self.ann.input_size):
            raise ValueError(f"Invalid input size. Expected: {self.ann.input_size}, got: {X.shape[1]}")
        #return
        return self.ann.predict(X)
    
    def test_backward(self):
        self.ann.backward_propagation()

    """
    Please change the validation part to use the validation set
    """
    def fit(self, X_train : np.array, Y_train : np.array, epochs : int, learning_rate : float, batch_size : int = 1, X_valid : np.array = None, Y_valid : np.array = None, error_threshold : float = 1e-5, ):
        # check if the last layer is output layer
        if self.ann.layers[-1].layer_type != "output":
            raise ValueError("The last layer must be output layer")
        if(len(X_train.shape) == 1):
            X_train = X_train.reshape(X_train.shape[0], 1)
        #if input shape invalid
        if(X_train.shape[1] != self.ann.input_size):
            raise ValueError(f"Invalid input size. Expected: {self.ann.input_size}, got: {X_train.shape[1]}")
        
        # configure ANN
        self.ann.learning_rate = learning_rate


        #train
        self.train_input = X_train
        self.train_output = Y_train

        for i in range(epochs):
            #looping for each batch
            iter = 0
            loss = 0
            for j in range(0, X_train.shape[0], batch_size):
                #get the batch
                X_batch = X_train[j:j+batch_size]
                Y_batch = Y_train[j:j+batch_size]
                self.ann.layers[-1].expected_output = Y_batch
                #forward propagation
                self.ann.forward_propagation(X_batch)
                #backward propagation
                self.ann.backward_propagation()
                loss += self.ann.get_current_loss()
                iter += 1

            avg_loss = loss / iter
            
            #break if the error is below the threshold
                        
            print(f"Epoch {i+1} completed")
            print(f"Loss: {avg_loss}")
            self.loses.append(avg_loss)
            self.accuracy(X_train, Y_train)
            # if(X_valid is not None and Y_valid is not None):
            #     self.accuracy(X_valid, Y_valid)

            if avg_loss < error_threshold:
                break

    def accuracy(self, X, Y):
        if(len(X.shape) == 1):
            X = X.reshape(X.shape[0], 1)
        #if input shape invalid
        if(X.shape[1] != self.ann.input_size):
            raise ValueError(f"Invalid input size. Expected: {self.ann.input_size}, got: {X.shape[1]}")
        
        #calculate the accuracy
        prediction = self.predict(X)
        correct = 0
        for i in range(len(prediction)):
            if np.argmax(prediction[i]) == np.argmax(Y[i]):
                correct += 1
        print(f"Accuracy: {correct/len(Y)*100}%")


        