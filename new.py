import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import os

df = pd.read_csv('house_prices.csv')
print(df.head())
print(df.isnull().sum())

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

X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
        'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
y = df['price']

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),
                                      ['mainroad', 'guestroom', 'basement',
                                       'hotwaterheating', 'airconditioning',
                                       'prefarea', 'furnishingstatus'])], remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

print("--- Enter your house preferences ---")
area = float(input("Enter the area of the house (in square feet): "))
bedrooms = int(input("Enter the number of bedrooms: "))
bathrooms = int(input("Enter the number of bathrooms: "))
stories = int(input("Enter the number of stories: "))
mainroad = input("Is the house on the main road? (yes/no): ")
guestroom = input("Is there a guestroom? (yes/no): ")
basement = input("Is there a basement? (yes/no): ")
hotwaterheating = input("Is there hot water heating? (yes/no): ")
airconditioning = input("Is there air conditioning? (yes/no): ")
parking = int(input("How many parking spaces are available? "))
prefarea = input("Is the house in a preferred area? (yes/no): ")
furnishingstatus = input("Furnishing status (furnished/semi-furnished/unfurnished): ")

user_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad, guestroom,
                           basement, hotwaterheating, airconditioning, parking,
                           prefarea, furnishingstatus]],
                         columns=['area', 'bedrooms', 'bathrooms', 'stories',
                                  'mainroad', 'guestroom', 'basement',
                                  'hotwaterheating', 'airconditioning',
                                  'parking', 'prefarea', 'furnishingstatus'])

try:
    user_data_encoded = ct.transform(user_data)
    predicted_price = regressor.predict(user_data_encoded)[0]
    print(f"Predicted Price for the given preferences: Rupees {predicted_price:,.2f}/-")

    columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
               'basement', 'hotwaterheating', 'airconditioning', 'parking',
               'prefarea', 'furnishingstatus', 'predicted_price']

    file_path = os.path.join(os.getcwd(), 'user_preferences.csv')
    file_exists = os.path.isfile(file_path)

    user_data['predicted_price'] = predicted_price
    user_data.to_csv(file_path, mode='a', header=not file_exists, index=False, columns=columns)
    print(f"Your preferences have been saved to '{file_path}'.")
except Exception as e:
    print(f"An error occurred: {e}")
