import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read your data
data = pd.read_csv("dataset/data.csv")

# Split the 'numbers' column into separate columns
numbers = data['numbers'].str.split(' ', expand=True)
numbers.columns = ['num1', 'num2', 'num3', 'num4']
numbers = numbers.astype(int)

# Combine the date and numbers columns
data = pd.concat([data['date'], numbers], axis=1)

# Add a numerical 'day' feature for better modeling
data['day'] = pd.to_datetime(data['date']).dt.dayofyear

# Prepare the features and target
X = data[['day']]  # Features (day of the year)
y = data[['num1', 'num2', 'num3', 'num4']]  # Target (numbers)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the next day's winning numbers
next_day = pd.DataFrame({'day': [data['day'].max() + 1]})
predicted_numbers = model.predict(next_day)

# Round the predicted numbers and ensure they are within 0-99 range
predicted_numbers = np.clip(np.round(predicted_numbers), 0, 99).astype(int)

# Display the predicted numbers
print(f'Predicted winning numbers for 2024/01/06: {" ".join(map(str, predicted_numbers[0]))}')
