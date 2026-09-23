import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dummy dataset: study hours -> exam score
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([35, 40, 50, 55, 62, 68, 74, 80, 85, 92])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R-squared: {r2_score(y, predictions):.4f}")

# Predict for a new value
new_hours = np.array([[11]])
print(f"Predicted score for 11 hours studied: {model.predict(new_hours)[0]:.2f}")