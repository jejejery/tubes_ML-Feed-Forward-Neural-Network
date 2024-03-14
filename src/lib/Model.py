from lib.ANN import ANN

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
        