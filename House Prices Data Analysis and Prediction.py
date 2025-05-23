import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('house_prices.csv')

# Display the first few rows to understand the data structure
print(df.head())

# Data Cleaning
# Check for missing values
print(df.isnull().sum())

# No missing values handling is needed if all data is present
# If there are missing values, handle them appropriately. Example:
#df['area'].fillna(df['area'].mean(), inplace=True)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='area', y='price', data=df)
plt.title('Area vs Price')
plt.xlabel('Area (square feet)')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.title('Prices by Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.show()

# Feature Engineering
# No additional features need to be engineered, but you can create new features if needed.

# Model Building
# Selecting features and target variable
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
        'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
y = df['price']

# Encoding categorical data (mainroad, guestroom, basement, etc.)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),
                                      ['mainroad', 'guestroom', 'basement',
                                       'hotwaterheating', 'airconditioning',
                                       'prefarea', 'furnishingstatus'])], remainder='passthrough')
X = ct.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()
