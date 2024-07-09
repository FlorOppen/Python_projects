import os
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

def inference():
    # Cargar los datos de prueba
    testing = "./test.csv"
    data_test = pd.read_csv(testing)
        
    y_test = data_test['target'].values
    X_test = data_test.drop(['target'], axis=1).values
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Cargar el escalador y normalizar los datos de prueba
    scaler = load('scaler.joblib')
    X_test = scaler.transform(X_test)
        
    # Cargar el modelo y realizar predicciones
    clf_nn = load('Inference_NN.joblib')
    print("NN score and classification:")
    print(clf_nn.score(X_test, y_test))
    print(clf_nn.predict(X_test))
    
if __name__ == '__main__':
    inference()
