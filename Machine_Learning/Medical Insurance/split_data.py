import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    # Load the original dataset
    data = pd.read_csv('insurance.csv')
    
    # Shuffle and split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    
    # Save the train and test sets to CSV files
    train_data.to_csv('train.csv', index=False)
    test_data.to_csv('test.csv', index=False)
    print("Data has been split and saved to train.csv and test.csv")

if __name__ == '__main__':
    split_data()