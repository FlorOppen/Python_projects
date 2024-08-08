
# Vendemos Todo Project

This project involves automating the process of managing product information and photos using Google Drive API and Python. It reads product data from an Excel file, filters available products, and handles their photos stored in Google Drive.

## Project Structure

The project includes the following files:

- **Vendemos_todo.ipynb**: Jupyter Notebook containing the code to process product data and manage photos using Google Drive API.
- **requirements.txt**: List of dependencies required to run the project.

## Overview

The main functionalities of this project are:

1. **Read Product Data**: Load product data from an Excel file.
2. **Filter Available Products**: Filter products based on their availability.
3. **Handle Product Photos**: Use Google Drive API to manage and retrieve product photos stored in Google Drive.

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

3. **Set up Google Drive API**:
    - Create a project on Google Cloud Platform.
    - Enable the Google Drive API.
    - Create credentials (OAuth 2.0 Client IDs) and download the `client_secrets.json` file.
    - Place the `client_secrets.json` file in the project directory.

4. **Run the Jupyter Notebook**:
    ```sh
    jupyter notebook Vendemos_todo.ipynb
    ```

## Libraries Used

- pandas
- pydrive
- fuzzywuzzy
- reportlab
- google-auth
- google-auth-oauthlib
- google-auth-httplib2
- googleapiclient

## Conclusion

This project demonstrates how to automate the management of product data and photos using Python and Google Drive API. It can be extended to include more complex operations or integrated into a larger system for e-commerce or inventory management.

## Acknowledgements

- The project utilizes various Python libraries and Google Drive API for its functionalities.

I hope you find this project interesting and useful. If you have any questions or suggestions, feel free to reach out.
