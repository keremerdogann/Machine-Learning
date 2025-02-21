import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("unvan_maas.csv")

print(df.columns)

X = df["Egitim Seviyesi"].values.reshape(-1,1)

Y = df["maas"].values.reshape(-1,1)

rfr_regressor = RandomForestRegressor(n_estimators=10,random_state=0)

rfr_regressor.fit(X,Y.ravel()) #buradaki RAVEL ın anlamını ögren

print("tahminler : ",rfr_regressor.predict(X))
print("gercek degerler: ",Y)

plt.title("GRAFİK")
plt.scatter(X,Y,color="red")
plt.plot(X-1.5,Y,color="yellow")
plt.plot(X+1.5,Y,color="blue")
plt.plot(X,rfr_regressor.predict(X))
plt.show()




