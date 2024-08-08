
# King County House Prices Project

This project aims to predict house prices in King County, which includes Seattle, using regression models. The dataset contains house sale prices for homes sold between May 2014 and May 2015.

## Dataset Description

The dataset contains information about 21,613 houses, with the following columns:

1. **id**: Unique identifier for each house.
2. **date**: Date the house was sold.
3. **price**: Sale price of the house.
4. **bedrooms**: Number of bedrooms.
5. **bathrooms**: Number of bathrooms.
6. **sqft_living**: Square footage of the living space.
7. **sqft_lot**: Square footage of the lot.
8. **floors**: Number of floors.
9. **waterfront**: Indicates if the house is on the waterfront.
10. **view**: Quality of the view from the house.
11. **condition**: Condition of the house.
12. **grade**: Overall grade given to the housing unit, based on King County grading system.
13. **sqft_above**: Square footage of the house apart from the basement.
14. **sqft_basement**: Square footage of the basement.
15. **yr_built**: Year the house was built.
16. **yr_renovated**: Year the house was renovated.
17. **zipcode**: ZIP code of the house location.
18. **lat**: Latitude coordinate.
19. **long**: Longitude coordinate.
20. **sqft_living15**: Living room area in 2015 (implies some renovations).
21. **sqft_lot15**: Lot area in 2015 (implies some renovations).

## Project Structure

The project includes the following files:

- **kc_house_exploration.ipynb**: Jupyter Notebook containing the exploratory data analysis (EDA) and model development.
- **app.py**: Flask application to serve the model.
- **Dockerfile**: Docker configuration for the Flask application.
- **docker-compose.yml**: Docker Compose configuration to set up the Flask app and MySQL database.
- **requirements.txt**: List of dependencies required to run the project.

## How to Run

1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Set up the environment variables**: Create a `.env` file with the following variables:
    ```env
    MYSQL_USER=<your_mysql_user>
    MYSQL_PASSWORD=<your_mysql_password>
    MYSQL_DB=<your_database_name>
    ```

3. **Build and run the containers**:
    ```sh
    docker-compose up --build
    ```

4. **Access the application**: The Flask application will be running on `http://localhost:5000`.

## Libraries Used

- Flask
- Scikit-learn
- Pandas
- Numpy
- Joblib
- MySQL Connector
- Python-dotenv

## Conclusion

The analysis and model development in this project provide insights into the factors affecting house prices in King County. The Flask application allows for easy deployment of the predictive model.

## Acknowledgements

- The dataset is publicly available on [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction).

I hope you find this project interesting and useful. If you have any questions or suggestions, feel free to reach out.
