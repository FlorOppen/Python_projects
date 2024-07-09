import pandas as pd
from joblib import load

def inference():
    # Load the test data
    data_test = pd.read_csv('test.csv')
    
    # Encode categorical variables in the same way as training data
    data_test['smoker'] = data_test['smoker'].map({'yes': 1, 'no': 0})
    data_test['sex'] = data_test['sex'].map({'female': 1, 'male': 0})
    data_test = pd.get_dummies(data_test, columns=['region'], drop_first=True)
    
    # Select the features for prediction
    X_test = data_test[['age', 'bmi', 'children', 'sex', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']]
    y_test = data_test['charges']
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)

    # Load the model pipeline and make predictions
    rf_model = load('rf_model.joblib')
    y_pred = rf_model.predict(X_test)
    
    # Print the model's score and predictions
    score = rf_model.score(X_test, y_test)
    print(f"Test Score: {score}")
    
if __name__ == '__main__':
    inference()