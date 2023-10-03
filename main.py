import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress FutureWarnings related to is_categorical_dtype
warnings.filterwarnings("ignore", category=FutureWarning)


# Load the data from the CSV file
data = pd.read_csv("tested.csv")

# Data Preprocessing
data = data.rename(columns=lambda x: x.strip())  # Remove leading/trailing spaces in column names
data['Year'] = data['Year'].str.extract('(\d+)').astype(float)  # Extract numeric part of "Year"
data['Duration'] = data['Duration'].str.extract('(\d+)').astype(float)  # Extract numeric part of "Duration"

# Encode categorical features (Director, Actor 1, Actor 2, Actor 3) using Label Encoding
label_encoders = {}
categorical_features = ["Director", "Actor 1", "Actor 2", "Actor 3"]
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# Convert the "Votes" column to numeric
data["Votes"] = data["Votes"].str.replace(',', '').astype(float)

# One-hot encode the "Genre" column
data = pd.get_dummies(data, columns=["Genre"], prefix=["Genre"])

# Drop the "Name" column as it is not useful for rating prediction
data.drop("Name", axis=1, inplace=True)

# Split the data into features and target
X = data.drop("Rating", axis=1)
y = data["Rating"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Advanced Model: XGBoost with Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
}

xgb_model = xgb.XGBRegressor()
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_

# Fit the best model
best_xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_xgb_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# You can use best_xgb_model to make predictions for new data
# Create a line graph to visualize actual vs. predicted ratings
plt.figure(figsize=(8, 6))
plt.plot(y_test, y_pred, 'o', label="Actual vs. Predicted")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', label="Perfect Alignment", color='red')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs. Predicted Movie Ratings")
plt.legend()
plt.grid(True)
plt.show()
# Plot feature importances
feature_importances = best_xgb_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names, orient='h', palette='viridis')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance Plot")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data["Votes"], bins=30, kde=True)
plt.xlabel("Number of Votes")
plt.ylabel("Frequency")
plt.title("Distribution of Votes for Movies")
plt.show()

genre_columns = [col for col in data.columns if col.startswith("Genre_")]
genre_counts = data[genre_columns].sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, orient='h', palette='Set3')
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.title("Distribution of Movie Genres")
plt.show()

top_directors = data["Director"].value_counts().head(10).index
director_avg_ratings = data.groupby("Director")["Rating"].mean().loc[top_directors]

plt.figure(figsize=(10, 6))
sns.barplot(x=director_avg_ratings.values, y=director_avg_ratings.index, orient='h', palette='Blues_d')
plt.xlabel("Average Rating")
plt.ylabel("Director")
plt.title("Average Ratings for Top Directors")
plt.show()

