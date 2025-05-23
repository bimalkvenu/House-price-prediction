The model is trained on a dataset (house_prices.csv) and evaluated using Mean Squared Error (MSE). The application also features a user-friendly terminal interface where users can input their house preferences and receive a predicted price. Additionally, user preferences and predictions are saved automatically into a CSV file for future reference.

✨ Features
📊 Exploratory Data Analysis (EDA)
Visualizations using Seaborn and Matplotlib to understand relationships between features and house prices:

Histogram of price distribution

Scatter plot of area vs price

Boxplot of price by number of bedrooms

🧠 Machine Learning Model

Trained a Linear Regression model to predict house prices

Preprocessing includes One-Hot Encoding for categorical features using ColumnTransformer

🧪 Model Evaluation

Performance evaluated with Mean Squared Error (MSE)

Visual comparison between actual and predicted prices using a scatter plot

🧾 Interactive Prediction Interface

Command-line interface for users to input house specifications

Predicts and prints the house price based on user input

💾 Data Logging

Stores user input and predicted price in a file named user_preferences.csv

🔧 Technologies Used
Python 3

Pandas

NumPy

Scikit-Learn

Matplotlib

Seaborn
