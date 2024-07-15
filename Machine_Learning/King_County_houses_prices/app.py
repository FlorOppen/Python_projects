from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load the environment variables
load_dotenv()

# Obtener las variables de entorno
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_DB = os.getenv("MYSQL_DB")

# Load the trained model pipeline
pipeline = joblib.load('rfr_model.joblib')

# Configure the MySQL connection
def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        if connection.is_connected():
            print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

# Create the table if it doesn't exist
def create_table(connection):
    if connection is None:
        print("Failed to create table: No connection to MySQL")
        return

    create_table_query = """
    CREATE TABLE IF NOT EXISTS model_queries (
        id INT AUTO_INCREMENT PRIMARY KEY,
        bedrooms INT,
        bathrooms FLOAT,
        sqft_living INT,
        sqft_lot INT,
        floors FLOAT,
        waterfront INT,
        view INT,
        house_condition INT,
        grade INT,
        sqft_above INT,
        sqft_basement INT,
        yr_built INT,
        yr_renovated INT,
        zipcode INT,
        latitude FLOAT,
        longitude FLOAT,
        year INT,
        month INT,
        prediction FLOAT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    cursor = connection.cursor()
    try:
        cursor.execute(create_table_query)
        connection.commit()
        print("Table 'model_queries' created successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

# Create the connection and table if it doesn't exist
connection = create_connection()
create_table(connection)

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        input_df = pd.DataFrame([input_data])
        
        # Convert date to year and month
        if 'date' in input_data:
            input_df['date'] = pd.to_datetime(input_df['date'])
            input_df['year'] = input_df['date'].dt.year
            input_df['month'] = input_df['date'].dt.month
            input_df = input_df.drop(['date'], axis=1)
        
        # Make the prediction using the pipeline
        log_prediction = pipeline.predict(input_df)[0]
        prediction = np.expm1(log_prediction)
        
        # Save the query to the database
        if connection is not None:
            cursor = connection.cursor()
            insert_query = """
            INSERT INTO model_queries (bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, house_condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, latitude, longitude, year, month, prediction)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                input_data.get('bedrooms'),
                input_data.get('bathrooms'),
                input_data.get('sqft_living'),
                input_data.get('sqft_lot'),
                input_data.get('floors'),
                input_data.get('waterfront'),
                input_data.get('view'),
                input_data.get('house_condition'),
                input_data.get('grade'),
                input_data.get('sqft_above'),
                input_data.get('sqft_basement'),
                input_data.get('yr_built'),
                input_data.get('yr_renovated'),
                input_data.get('zipcode'),
                input_data.get('latitude'),        
                input_data.get('longitude'),       
                input_data.get('year'),
                input_data.get('month'),
                prediction
            ))
            connection.commit()
        else:
            print("Failed to insert data: No connection to MySQL")
        
        return jsonify({'prediction': prediction})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)