# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries (numpy, pandas, sklearn)
2. Load the dataset (house features like area, rooms, location, etc.)
3. Separate input features (X) and outputs (y → price & occupants)
4. Normalize or standardize the feature values using scaling
5. Convert data into appropriate format for model
6. Import SGD Regressor from sklearn.linear_model
7. Fit the SGD model using training data
8. Repeat until convergence or max iterations reached
9. Compare predicted values with actual values
10. Calculate accuracy metrics (MSE, R² score)

## Program:


```

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: K MONISHWAR
RegisterNumber:  212225230188

CODING:

import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("Downloads\house.csv")
data.columns = data.columns.str.strip()
X = data[['Size', 'Bedrooms']]
y_price = data['Price']
y_occ = data['Occupants']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)
size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))
new_data = scaler.transform([[size, bed]])

pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```

## Output:

<img width="1181" height="766" alt="image" src="https://github.com/user-attachments/assets/cd3b383b-e384-462c-943e-a7543c6bb45b" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
