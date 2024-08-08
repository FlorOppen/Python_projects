
# Online Retail Analysis Project

This project aims to analyze the impact of a discount strategy on product sales using the "Online Retail" dataset. The dataset contains transactional data from an online retail company based in the United Kingdom, spanning from December 2010 to December 2011.

## Dataset Description

The dataset includes information about the invoices, products, quantities, prices, and customer details. The columns in the dataset are:

- **InvoiceNo**: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction.
- **StockCode**: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
- **Description**: Product (item) name. Nominal.
- **Quantity**: The quantities of each product (item) per transaction. Numeric.
- **InvoiceDate**: Invoice date and time. Numeric, the day and time when a transaction was generated.
- **UnitPrice**: Unit price. Numeric, Product price per unit in sterling.
- **CustomerID**: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
- **Country**: Country name. Nominal, the name of the country where each customer resides.

## Project Structure

The project includes the following files:

- **Online_Retail.ipynb**: Jupyter Notebook containing the data exploration, cleaning, and statistical analysis.

## Data Exploration and Cleaning

The initial data exploration and cleaning steps include:

1. **Handling Missing Values**:
    - Columns with missing values: `Description` and `CustomerID`.
    - Rows with missing `Description` also have missing `CustomerID`.
    - `CustomerID` has almost 25% missing values.
    - All rows with missing `CustomerID` and `Description` are dropped.

2. **Data Types**:
    - Columns: 5 object, 2 float, and 1 int64.
    - `CustomerID` converted from float to object.

3. **Negative Values**:
    - Check for negative values in `Quantity` and `UnitPrice`.
    - Handle or clean any negative or zero values appropriately.

## Goal of the Analysis

The primary goal of this analysis is to evaluate the effectiveness of a discount strategy on product sales. The analysis includes statistical techniques to determine the impact of discounts on sales performance.

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

3. **Run the Jupyter Notebook**:
    ```sh
    jupyter notebook Online_Retail.ipynb
    ```

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Conclusion

This project demonstrates the use of statistical techniques to analyze the impact of discount strategies on online retail sales. It provides insights into the effectiveness of discounts and helps in making data-driven decisions.

## Acknowledgements

- The dataset is publicly available on the UCI Machine Learning Repository.

I hope you find this project interesting and useful. If you have any questions or suggestions, feel free to reach out.
