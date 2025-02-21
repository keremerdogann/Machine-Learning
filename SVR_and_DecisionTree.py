import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import statsmodels.api as sm
from sklearn.svm import SVR

df = pd.read_csv("unvan_maas.csv")

X = df["Egitim Seviyesi"].values.reshape(-1,1)
Y = df["maas"].values.reshape(-1,1)


## SVR İÇİN OLCEKLENDİRME ONEMLİDİR , DİGER TÜRLÜ CIKTILAR UYGUN OLMAYABİLİR !
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

svr_reg = SVR(kernel="rbf") # kernel kısmına farklı metodlar gelebilir
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli)
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))
plt.show()

print("destek vektor tahminleri")
print(f"tahmin: {svr_reg.predict(x_olcekli)}")
print(y_olcekli)


from sklearn.tree import DecisionTreeRegressor

t_regressor = DecisionTreeRegressor(random_state=0)
t_regressor.fit(X,Y)

plt.scatter(X,Y)
plt.plot(X,t_regressor.predict(X))
plt.show()

print("karar ağacı sonucları")
print(f"tahmin : {t_regressor.predict(X)}")
print(Y)
print(t_regressor.predict([[1]]))
print(t_regressor.predict([[6.6]]))



