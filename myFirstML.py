import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Load in the data and save it as a DataFrame
df = pd.read_csv('auto-mpg.csv')

# Print out the column names
print('Column names are: ', list(df.columns))

pd.scatter_matrix(df, alpha=0.4, figsize=(7, 7))
plt.show()

# Make target (y) equal to mpg
y = df.pop('mpg')

# Make x a large matrix containing displacement, cylinders, weight, acceleration and model year
X = df[['displacement', 'cylinders', 'weight', 'acceleration', 'model year']]

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initilize the object
reg = LinearRegression()

# Train on the training data
reg.fit(X_train, y_train)

# Predict on the test X set
predictions = reg.predict(X_test)

print(list(zip(X_train.columns, reg.coef_)))

mean_error = np.sqrt(np.mean((y_test - predictions) ** 2))

print('On Average we were: ', np.round(mean_error), ' off from the actual mpg')


def predict(displacement, cylinders, weight, acceleration, model_year):

    # Create Predictions
    prediction = reg.predict([[displacement, cylinders, weight, acceleration, model_year]])

    print("Your Predicted MPG is: ", np.round(prediction[0]))


displacement = 193
cylinders = 6
weight = 2900
acceleration = 15
model_year = 76

predict(displacement, cylinders, weight, acceleration, model_year)
