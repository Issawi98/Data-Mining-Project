import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def split(self, X, y, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 58) #58
        return X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64)
    
    def splitCluster(self, X, test_size):
        X_train, X_test = train_test_split(X, test_size = test_size)
        return X_train, X_test
    
    def scale(self, X_train, X_test, type):
        if type == "StandardScaler":
            standardScaler = StandardScaler()
            standardScaler.fit(X_train)
            X_train = standardScaler.transform(X_train)
            X_test = standardScaler.transform(X_test)
            return X_train, X_test
        
        elif type == "MinMaxScaler":
            minMaxScaler = MinMaxScaler()
            minMaxScaler.fit(X_train)
            X_train = minMaxScaler.transform(X_train)
            X_test = minMaxScaler.transform(X_test)
            return X_train, X_test
        elif type == "MaxScaler":
            
            maxScaler = MaxAbsScaler()
            maxScaler.fit(X_train)
            X_train = maxScaler.transform(X_train)
            X_test = maxScaler.transform(X_test)
            return X_train, X_test
        
        elif type == "RobustScaler":
            robustScaler = RobustScaler()
            robustScaler.fit(X_train)
            X_train = robustScaler.transform(X_train)
            X_test = robustScaler.transform(X_test)
            return X_train, X_test
        
    def encode(self, X, y, dataset_name):
        if dataset_name == 'diamonds':
            labelencoder = LabelEncoder()
            X[:,1] = labelencoder.fit_transform(X[:,1])
            X[:,2] = labelencoder.fit_transform(X[:,2])
            X[:,3] = labelencoder.fit_transform(X[:,3])
        elif dataset_name == 'cancer':
            labelencoder = LabelEncoder()
            y[:,0] = labelencoder.fit_transform(y[:,0])
        return X, y

    def drop(self, data):
        data_modified =  data.dropna(axis=0)
        return data_modified
    
    def replaceMean(self, data):
        data_modified =  data.fillna(data.mean(), inplace=True)
        return data_modified
    
    def replaceMode(self, data):
        data_modified =  data.fillna(data.mode(), inplace=True)
        return data_modified
        
    def dataCleaning(self, data, name):
        if name == 'cancer':
            data.drop(data.columns[0], axis='columns', inplace = True)
            X = data.iloc[:, 1:].to_numpy()
            y = data.iloc[:, 0].values.reshape(569,1)
                
        elif name == 'diamonds':
            data.drop(data.columns[0], axis='columns', inplace = True)
            data_modified = data.drop(labels='price', axis=1)
            X = data_modified.iloc[:, 0:8].to_numpy()
            y = data.iloc[:, 6].values.reshape(53940, 1)
        return X,y
    