
# Medical Insurance Cost Prediction Project

This project aims to analyze medical insurance data and build a predictive model for insurance costs.

## Dataset Description

The dataset contains information about individuals and their medical insurance costs. The features include age, sex, BMI, number of children, smoking status, and region. The target variable is the insurance charges.

## Project Structure

The project includes the following files:

- **Medical_Insurance.ipynb**: Jupyter Notebook containing the data exploration, preprocessing, and model development.
- **train.py**: Script to train the machine learning model.
- **split_data.py**: Script to split the dataset into training and testing sets.
- **inference.py**: Script to perform inference using the trained model.
- **app.py**: Flask application to serve the model.
- **Dockerfile**: Docker configuration for the Flask application.
- **requirements.txt**: List of dependencies required to run the project.
- **train.csv**: Training data.
- **test.csv**: Testing data.

## Data Preparation

The following steps are involved in data preparation:

1. **Train-Test Split**: Separating the data into training and testing sets.
2. **Scaling**: Scaling the data using appropriate techniques.
3. **Encoding**: Converting categorical features into numerical values.

## Model Building

The model is built using various regression techniques to predict insurance costs. The following steps are involved:

1. **Data Exploration**: Analyzing the dataset to understand the distribution and relationships between features.
2. **Feature Engineering**: Creating new features and transforming existing ones to improve model performance.
3. **Model Training**: Training the model using the training dataset.
4. **Model Evaluation**: Evaluating the model using appropriate metrics.

## How to Run

1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the data split script**:
    ```sh
    python split_data.py
    ```

4. **Train the model**:
    ```sh
    python train.py
    ```

5. **Run the Flask application**:
    ```sh
    python app.py
    ```

6. **Access the application**: The Flask application will be running on `http://localhost:5000`.

## Libraries Used

- Flask
- Scikit-learn
- Pandas
- Numpy
- Joblib
- Matplotlib
- Seaborn

## Conclusion

The project demonstrates the use of regression techniques to predict medical insurance costs. The Flask application allows for easy deployment and inference using the trained model.

## Acknowledgements

- The dataset is publicly available on Kaggle.

I hope you find this project interesting and useful. If you have any questions or suggestions, feel free to reach out.
