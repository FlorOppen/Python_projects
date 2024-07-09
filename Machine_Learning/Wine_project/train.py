import os
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

def train():
    # Cargar los datos de entrenamiento
    training = "./train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['target'].values
    X_train = data_train.drop(['target'], axis=1).values

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
        
    # Normalizar los datos usando StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Entrenar el modelo
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=10000)
    clf_NN.fit(X_train, y_train)
    
    # Imprimir el puntaje del modelo en el conjunto de entrenamiento
    score = clf_NN.score(X_train, y_train)
    print(f"Training Score: {score}")
       
    # Guardar el modelo y el escalador
    dump(clf_NN, 'Inference_NN.joblib')
    dump(scaler, 'scaler.joblib')
        
if __name__ == '__main__':
    train()


