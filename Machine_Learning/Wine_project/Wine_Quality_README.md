
# Wine Quality Prediction Project

This project aims to predict the quality of wine using various machine learning models.

## Dataset Description

The dataset contains information about various physicochemical properties of wine and their corresponding quality ratings. The features include fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and quality.

## Project Structure

The project includes the following files:

- **train.py**: Script to train the machine learning model.
- **inference.py**: Script to perform inference using the trained model.
- **download_dataset.py**: Script to download the dataset.
- **app.py**: Flask application to serve the model.
- **Dockerfile**: Docker configuration for the Flask application.
- **requirements.txt**: List of dependencies required to run the project.
- **Inference_NN.joblib**: Trained model file.

## Data Preparation

The following steps are involved in data preparation:

1. **Download Dataset**: Download the dataset using `download_dataset.py`.
2. **Train-Test Split**: Separating the data into training and testing sets.
3. **Scaling**: Scaling the data using appropriate techniques.
4. **Encoding**: Converting categorical features into numerical values.

## Model Building

The model is built using various machine learning techniques to predict wine quality. The following steps are involved:

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

3. **Download the dataset**:
    ```sh
    python download_dataset.py
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

The project demonstrates the use of machine learning techniques to predict wine quality. The Flask application allows for easy deployment and inference using the trained model.

## Acknowledgements

- The dataset is publicly available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).

I hope you find this project interesting and useful. If you have any questions or suggestions, feel free to reach out.
