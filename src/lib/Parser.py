import json
import numpy as np
from lib.HiddenLayer import HiddenLayer
from lib.OutputLayer import OutputLayer
from lib.Model import Model
from lib.ANN import ANN

class Parser:
    def __init__(self, file_path):
        data = 0
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        self.case = data["case"]
        self.input = np.array(self.case["input"])
        try:
            self.weights = list(self.case["weights"])
        except:
            print()

        model = self.case["model"]
        self.input_size = model["input_size"]
        self.layers = model["layers"]

        self.expect = data["expect"]
        try:
            self.expected_output = np.array(self.expect["output"])
        except:
            print()
        try:
            self.max_sse = self.expect["max_sse"]
        except:
            print()

    def getInputSize(self):
        return self.input_size
    
    def getOutputSize(self):
        return self.layers[-1]["number_of_neurons"]

    def addAllLayers(self, model, excpected_output = None):
        layers_size = len(self.layers)
        for i in range(layers_size):
            if i == 0:
                layer = HiddenLayer(name=f"hidden{i+1}", input_shape=self.input_size, output_shape=self.layers[i]["number_of_neurons"], weights=np.array(self.weights[i]), activation_function=self.layers[i]["activation_function"])
                model.add(layer)
            elif i!=(layers_size-1):
                layer = HiddenLayer(name=f"hidden{i+1}", input_shape=self.layers[i-1]["number_of_neurons"], output_shape=self.layers[i]["number_of_neurons"], weights=np.array(self.weights[i]), activation_function=self.layers[i]["activation_function"])
                model.add(layer)
            else:
                layer = OutputLayer(name="output1",input_shape=self.layers[i-1]["number_of_neurons"], output_shape=self.layers[i]["number_of_neurons"], weights=np.array(self.weights[i]), activation_function=self.layers[i]["activation_function"], expected_output=excpected_output)
                model.add(layer)

    def getExpectedOutput(self):
        return np.array(self.expected_output)
    
    def getMaxSse(self):
        return self.max_sse
    
    def getSse(self,prediction):
        res = self.getExpectedOutput() - prediction
        res = res**2
        #flatten the array
        res = res.flatten()
        return np.sum(res)

    def isCorrect(self, prediction):
        return self.getSse(prediction) <= self.getMaxSse()
        
    
class BackPropParser(Parser): 
    def __init__(self, file_path):
        super().__init__(file_path)

        self.weights = list(self.case["initial_weights"])
        self.target = np.array(self.case["target"])
        
        learning_parameters = self.case["learning_parameters"]

        self.learning_rate = learning_parameters["learning_rate"]
        self.batch_size = learning_parameters["batch_size"]
        self.max_iteration = learning_parameters["max_iteration"]
        self.error_threshold = learning_parameters["error_threshold"]

        self.stopped_by = self.expect["stopped_by"]
        self.final_weights = list(self.expect["final_weights"])

    def build_model(self) -> Model :
        model = Model("the_model", ANN(self.input_size,len(self.target[0]), self.learning_rate, self.max_iteration, self.error_threshold))
        self.addAllLayers(model, self.target)
        return model
        