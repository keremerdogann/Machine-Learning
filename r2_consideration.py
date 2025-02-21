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
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("unvan_maas.csv")

X = df["Egitim Seviyesi"].values.reshape(-1,1)
Y = df["maas"].values.reshape(-1,1)

#sc=StandardScaler()

#x_olcekli=sc.fit_transform(X)
#y_olcekli=sc.fit_transform(Y)

lr = LinearRegression()
lr.fit(X,Y)

gercek_dataframe = pd.DataFrame(Y,columns=["Gercek Degerler"])
tahmin_dataframe = pd.DataFrame(lr.predict(X),columns=["Tahmin Degerleri"])

gercek_tahmin = pd.concat([gercek_dataframe,tahmin_dataframe],axis=1)

print("Lineer regresyon gercek ve tahmin degerleri :")
print(gercek_tahmin)

print("Lineer regresyon R2 degeri")
print(r2_score(Y,lr.predict(X)))

poly_reg = PolynomialFeatures(degree=4)

x_poly = poly_reg.fit_transform(X)

print(f"x_poly degeri : {x_poly}")

lr2 = LinearRegression()

lr2.fit(x_poly,Y)

gercek_dataframe = pd.DataFrame(Y,columns=["Gercek Degerler"])
tahmin_dataframe = pd.DataFrame(lr2.predict(x_poly),columns=["Tahmin Degerleri"])

gercek_tahmin = pd.concat([gercek_dataframe,tahmin_dataframe],axis=1)


print("Polinomal regresyon gercek ve tahmin degerleri :")
print(gercek_tahmin)

print("Polinomal regresyon R2 degeri")
print(r2_score(Y,lr2.predict(x_poly)))

# simdi destek vektorle tahmin yapacağız ancak bunun için standartlastırma yapacağımız için gerçek ve tahmin değerleri
# arasındaki farkları gözle karar vermek dogru olmayacağından sadece r2 degerini inceleyecegim.

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

svr_reg =SVR(kernel="rbf") #rbf kısmına farklı metodlar gelebilir
svr_reg.fit(x_olcekli,y_olcekli)

print("Destek vektor r2 degeri")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))


#KARAR AĞACINDA R2 DEGERİ 1 CIKIYOR FAKAT BU BİZİ YANILTIR ÇÜNKÜ BİLİYORUZ Kİ KARAR AGACI BELİRLİ ARALIKTA HEP AYNI DEGERİ ALIYOR


t_regressor = RandomForestRegressor()

t_regressor.fit(X,Y)

gercek_dataframe = pd.DataFrame(Y,columns=["Gercek Degerler"])
tahmin_dataframe = pd.DataFrame(t_regressor.predict(X),columns=["Tahmin Degerleri"])

gercek_tahmin = pd.concat([gercek_dataframe,tahmin_dataframe],axis=1)

print(f"Rassal ormanlar gercek ve tahmin degerleri : {gercek_tahmin}")
print(f"Rassal ormanın r2 skor degeri : {r2_score(Y,t_regressor.predict(X))}")
























