
import numpy as np
from lib.HiddenLayer import HiddenLayer
from lib.OutputLayer import OutputLayer

class ANN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.layers = []
        self.output_size = output_size

    def add(self, layer):
        if isinstance(layer, HiddenLayer):
            self.layers.append(layer)
        elif isinstance(layer, OutputLayer):
            self.layers.append(layer)
        else:
            raise ValueError("Invalid layer type. Only HiddenLayer or OutputLayer allowed.")

    def debug(self):
        for layer in self.layers:
            print(f"Layer: {layer.name} - {layer.layer_type}")