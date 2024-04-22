import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PreProcessing:
    def __init__(self):
        self.file_path = None
        self.data_frame = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()

    def readCSV(self,file_path):
        self.file_path = file_path
        self.data_frame = pd.read_csv(file_path)     

    def dropTarget(self,column_name):
        self.X = self.data_frame.drop(column_name, axis=1)
        self.y = pd.get_dummies(self.data_frame[column_name])

    def splitTrainValid(self, valid_size = 0.25, random_state = 45):
        return train_test_split(self.X, self.y, test_size=valid_size, random_state=random_state)

    def fit_transform(self,X_train):
        return self.scaler.fit_transform(X_train)
    
    def transform(self,X_test):
        return self.scaler.transform(X_test)
