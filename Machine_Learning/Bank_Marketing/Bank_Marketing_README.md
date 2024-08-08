
# Bank Marketing Project

This project analyzes the direct marketing campaigns of a Portuguese banking institution. The primary focus is on assessing the effectiveness of their marketing campaigns, which were predominantly conducted through phone calls. These campaigns aimed at promoting the bank's term deposit products.

## Dataset Description

The dataset comprises 41,188 instances and 20 attributes, including the target variable (y). The features are both numeric and categorical, and there are no null values.

## Data Comprehension

- **Occupations**: Predominantly administrative, blue-collar, and technical.
- **Marital Status**: Mostly married individuals.
- **Education**: Primarily those with secondary or complete university education.
- **Default Credits**: No individuals with default credits.
- **Loans**: The majority do not have a loan.
- **Contact Month**: Most contacts occurred in May.
- **Previous Campaigns**: Most clients did not participate in previous campaigns (poutcome).
- **Target Variable**: The majority rejected the term deposit (unbalanced target).

## Exploratory Data Analysis (EDA)

We observed the following patterns during our EDA:

1. **Client Occupations**: A significant portion of the clients are engaged in administrative, blue-collar, and technical jobs.
2. **Marital Status and Education**: Married individuals and those with secondary or complete university education are prevalent.
3. **Loan Status**: The majority of clients do not have any loans.
4. **Contact Month**: The peak of the marketing campaigns occurred in May.
5. **Previous Campaign Outcomes**: Most clients had not participated in previous campaigns.
6. **Term Deposit Subscription**: The dataset is unbalanced with a higher number of clients rejecting the term deposit offer.

## Model Building and Evaluation

1. **Data Preprocessing**: Steps included encoding categorical variables, feature scaling, and handling class imbalance.
2. **Model Selection**: Various classification models were considered, including logistic regression, decision trees, and random forests.
3. **Performance Metrics**: Evaluated using metrics like accuracy, precision, recall, and F1-score.
4. **Results**: The models were compared, and the best-performing model was selected based on the evaluation metrics.

## Conclusion

The analysis provided insights into the client demographics and their responses to the marketing campaigns. The best-performing model could help the bank in targeting the right clients for their term deposit products in future campaigns.

## How to Run

1. Clone the repository.
2. Ensure you have all the required libraries installed.
3. Run the Jupyter notebook `Bank_Marketing_Project.ipynb` to see the analysis and results.

## Libraries Used

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Acknowledgements

- The dataset is publicly available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

I hope you find this project interesting and useful. If you have any questions or suggestions, feel free to reach out.
