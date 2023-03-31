import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv('data.csv')
data.head()                                                          

X = data[[ 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY', 
 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']]
Y = data['CO2EMISSIONS']

train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.8))):]

regr = linear_model.LinearRegression()
train_x = np.array(train[[ 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']])
train_y = np.array(train['CO2EMISSIONS'])
regr.fit(train_x,train_y)
test_x = np.array(test[[ 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']])
test_y = np.array(test['CO2EMISSIONS'])

coeff_data = pd.DataFrame(regr.coef_ , X.columns , columns=['Coefficients'])
coeff_data

Y_pred = regr.predict(test_x)

from sklearn.metrics import r2_score
R = r2_score(test_y , Y_pred)
print ('R² :',R)