import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#create instance for the dataset url and read in the dataset using pandas
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

#convert independent and dependent variables to numpy arrays and reshape independent variable array.
x = df.ENGINESIZE.to_numpy()
y = df.CO2EMISSIONS.to_numpy()
x = x.reshape(-1, 1)

#import module needed to split data for training and testing
from sklearn.model_selection import train_test_split

#x and y arrays are split 80% train to 20% test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#create instance for linear regression model and train
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

#create prediction instance and test against our y test values. Use mean squared and r2 scores to evaluate prediction results.
CO2EmissionPrediction = regressor.predict(X_test)
Difference = mean_squared_error(y_test, CO2EmissionPrediction)
Fit = r2_score(y_test, CO2EmissionPrediction)



print('Coefficient: \n', regressor.coef_)
print('Intercept: \n', regressor.intercept_)
print('The average squared difference between the predicted value and test value is: \n', Difference)
print(f'This means that the observed CO2 Emissions at a given Engine Size are on average {round(np.sqrt(Difference), 2)} g/mile below or above the predicted value.')
print()
print('The R2 score of this model is: \n', Fit)
print(f'This means that {round(100*Fit, 2)} of the variation in CO2 Emissions can be explained by the Engine size in our regression model.')
if Fit < .19:
    print('Therefore our model has a very weak fit for explaining CO2 emissions as they relate to engine size.')
elif .2 < Fit < .39:
    print('Therefore our model has a weak fit for explaining CO2 emissions as they relate to engine size.')
elif .4 < Fit < .59:
    print('Therefore our model has a medium fit for explaining CO2 emissions as they relate to engine size.')
elif .6 < Fit < .79:
    print('Therefore our model has a strong fit for explaining CO2 emissions as they relate to engine size.')
else:
    print('Therefore our model has a very strong fit for explaining CO2 emissions as they relate to engine size.')
