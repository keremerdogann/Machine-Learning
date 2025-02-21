import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv("unvan_maas.csv")

print(df)

egitim_seviyesi = df["Egitim Seviyesi"].values.reshape(-1,1)

maas = df["maas"]

#x_train , x_test , y_train , y_test = train_test_split(egitim_seviyesi,maas,test_size=0.33,random_state=0)

regressor = LinearRegression()

regressor.fit(egitim_seviyesi,maas.values)

y_pred = regressor.predict(egitim_seviyesi)

print(y_pred)
print(maas)

plt.scatter(egitim_seviyesi,maas.values)

plt.plot(egitim_seviyesi,regressor.predict(egitim_seviyesi))

plt.show()


from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4) #derece arttıkça tahmin gücü de artıyor / her zaman değil

x_poly = poly_reg.fit_transform(egitim_seviyesi)

print(x_poly)

regressor2 = LinearRegression()

regressor2.fit(x_poly,maas)

plt.scatter(egitim_seviyesi,maas)

plt.plot(egitim_seviyesi,regressor2.predict(poly_reg.fit_transform(egitim_seviyesi)))

plt.show()

print(regressor2.predict(poly_reg.fit_transform(egitim_seviyesi)))
print(maas)














