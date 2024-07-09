import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

def train():
    # Load the dataset
    data = pd.read_csv('train.csv')

    # Encode categorical variables
    data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
    data['sex'] = data['sex'].map({'female': 1, 'male': 0})
    data = pd.get_dummies(data, columns=['region'], drop_first=True)
    
    # Split the data into features and target
    X = data.drop(columns=['charges'])
    y = data['charges']
    
    # Define the preprocessing steps for numerical and categorical features
    numerical_features = ['age', 'bmi', 'children']
    categorical_features = [col for col in X.columns if col not in numerical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', 'passthrough', categorical_features)
        ])
    
    # Create a pipeline with the preprocessor and the model
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    print(f"Average cross-validation score: {cv_scores.mean()}")
    
    # Train the model on the full training data
    rf_model.fit(X, y)
    
    # Save the model and the preprocessor
    dump(rf_model, 'rf_model.joblib')

if __name__ == '__main__':
    train()