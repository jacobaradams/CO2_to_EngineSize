import numpy as np
from EmissionPredictionModel import regressor as rg


print()
print()


depen = np.array(float(input('What is the engine size you are looking at?')))
depen = depen.reshape(-1,1)

new = rg.predict(depen)

for i in new:
    print(f'Based on this model, an engine size of {depen[0][0]} liters has an average CO2 emission of {round(i, 2)} g/year! ')