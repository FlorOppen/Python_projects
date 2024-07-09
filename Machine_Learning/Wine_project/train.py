import os
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def train():
    # Load the training data
    training = "./train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['target'].values
    X_train = data_train.drop(['target'], axis=1).values

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
        
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Define the model
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=10000)
    
    # Perform cross-validation
    cv_scores = cross_val_score(clf_NN, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores)}")
    
    # Train the model on the entire dataset
    clf_NN.fit(X_train, y_train)
    
    # Print the model's score on the training set
    score = clf_NN.score(X_train, y_train)
    print(f"Training Score: {score}")
       
    # Save the model and the scaler
    dump(clf_NN, 'Inference_NN.joblib')
    dump(scaler, 'scaler.joblib')
        
if __name__ == '__main__':
    train()
