import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy import sqrt


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

model = LinearRegression()

x_train = train.drop('SalePrice', axis=1) # attrbutes so called x
y_train = train.loc[:,'SalePrice'] #label so called y

model.fit(x_train, y_train) # its applying our data into model

x_test = test.drop('SalePrice', axis=1)
y_test = test.loc[:,'SalePrice']

predictions = model.predict(x_test) 

comparison = pd.DataFrame({"Actual Values": y_test, "Predicted Values": predictions})


rmse = sqrt(mean_squared_error(y_test, predictions))

correlations = train.corr()

saleprice_correlations = correlations["SalePrice"]

saleprice_correlations = saleprice_correlations.sort_values(ascending = False) # ascending supplies sort from highest to lowest


