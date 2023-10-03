# Movie-Rating-Prediction-CodSoft
A Python script that performs data analysis, preprocessing, model training, and evaluation to predict the survival of passengers on the Titanic. The script uses the Random Forest Classifier to make predictions.

Prerequisites
Before running the code, make sure you have the following dependencies installed:
pandas
numpy
matplotlib
seaborn
scikit-learn

You can install these dependencies using pip:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
Usage

Clone this repository:
bash
Copy code
git clone https://github.com/your-username/titanic-survival-prediction.git

Navigate to the project directory:
bash
Copy code
cd titanic-survival-prediction
Download the Titanic dataset as a CSV file and place it in the project directory with the name tested.csv.

Run the Python script:
bash
Copy code
python titanic_survival_prediction.py
Overview
Data Loading: The script begins by loading the Titanic dataset from the tested.csv file using pandas.

Data Preprocessing:

Rows with missing values are dropped from the dataset.
Feature engineering is performed to extract titles from passenger names and create a new feature 'Title'.
'Title' values are categorized into common categories (e.g., 'Mr', 'Miss', 'Mrs', 'Master', 'Rare').
A new feature 'FamilySize' is created by adding 'SibSp' and 'Parch'.
Unnecessary columns ('PassengerId', 'Name', 'Ticket', 'Cabin', 'Age') are dropped from the dataset.
Categorical variables are encoded using one-hot encoding.
Data Splitting: The data is split into training and testing sets using train_test_split from scikit-learn.

Feature Scaling: Numeric features ('Fare' and 'FamilySize') are standardized using StandardScaler.

Model Training:

A Random Forest Classifier is initialized.
Hyperparameter tuning is performed using GridSearchCV to find the best combination of hyperparameters.
Model Evaluation:

The best model is used to make predictions on the test set.
Classification report and confusion matrix are printed to evaluate the model's performance.
Visualizations:

Several visualizations are created using seaborn and matplotlib to explore the data:
Survival count by gender
Survival rate by passenger class
Survival rate by title
