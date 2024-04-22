import json
import numpy as np
from lib.HiddenLayer import HiddenLayer
from lib.OutputLayer import OutputLayer

class Parser:
    def __init__(self, file_path):
        data = 0
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        case = data["case"]
        self.input = np.array(case["input"])
        self.weights = list(case["weights"])

        model = case["model"]
        self.input_size = model["input_size"]
        self.layers = model["layers"]

        expect = data["expect"]
        self.expected_output = np.array(expect["output"])
        self.max_sse = expect["max_sse"]

    def getInputSize(self):
        return self.input_size
    
    def getOutputSize(self):
        return self.layers[-1]["number_of_neurons"]

    def addAllLayers(self, model):
        layers_size = len(self.layers)
        for i in range(layers_size):
            if i == 0:
                layer = HiddenLayer(name=f"hidden{i+1}", input_shape=self.input_size, output_shape=self.layers[i]["number_of_neurons"], weights=np.array(self.weights[i]), activation_function=self.layers[i]["activation_function"])
                model.add(layer)
            elif i!=(layers_size-1):
                layer = HiddenLayer(name=f"hidden{i+1}", input_shape=self.layers[i-1]["number_of_neurons"], output_shape=self.layers[i]["number_of_neurons"], weights=np.array(self.weights[i]), activation_function=self.layers[i]["activation_function"])
                model.add(layer)
            else:
                layer = OutputLayer(name="output1",input_shape=self.layers[i-1]["number_of_neurons"], output_shape=self.layers[i]["number_of_neurons"], weights=np.array(self.weights[i]), activation_function=self.layers[i]["activation_function"], expected_output=None)
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
    
        
    
