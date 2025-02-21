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

label_encoder = LabelEncoder()

df = pd.read_csv("unvan_maas_odev.csv")

print("hicbir degisiklik yapılmamıs dataframe")
print(df)

df.set_index(["Calisan_ID"],inplace=True)

"""

unvan_order = {
    'Cayci': 1,
    'Sekreter': 2,
    'Uzman Yardimcisi': 3,                #UNVANLARA 1-2-3 DİYE AYIRDIK AMA UNVAN SEVİYESİ KOLONUNDA ZATEN BUNLAR YAZIYOR
    'Uzman': 4,
    'Proje Yoneticisi': 5,
    'Sef': 6,
    'Mudur': 7,
    'Direktor': 8,
    'C-level': 9,
    'CEO': 10
}

df['Unvan'] = df['Unvan'].map(unvan_order)

"""
print(df.columns)

Y = df["maas"].values.reshape(-1,1)
X = df.drop(columns=["maas","Unvan"])

print("unvan ve maas cıkarıldı")
print(X)


#poly_reg = PolynomialFeatures(degree=2)
#x_poly = poly_reg.fit_transform(X)

lr = LinearRegression()
lr.fit(X,Y)

sabit = sm.add_constant(X)

model = sm.OLS(Y,sabit).fit()
print("\nStatsmodels OLS Model Özeti")
print(model.summary())


#X = X.drop(columns=["Kidem","Puan"])

#lr.fit(X,Y)


#sabit = sm.add_constant(X)
#model = sm.OLS(lr.predict(X),X)
#print("\nStatsmodels OLS Model Özeti")
#print(model.fit().summary())

print("10 yıl tecrübeli ve 100 puan almış CEO nun tahmini maaşı :")

ceo =np.array([[10,10,100]])

print(lr.predict(ceo))

print("10 yıl tecrübeli ve 100 puan almış Müdür ün tahmini maaşı:")

mudur = np.array([[7,10,10]])

print(lr.predict(mudur))













