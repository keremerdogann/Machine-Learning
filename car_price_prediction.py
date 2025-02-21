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


df = pd.read_csv("audi.csv")


#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)







print(f"   \n \n ---------------- İLK HALİ --------------------------------")
print(df)
print(df.columns)
print(df["model"].unique())
print(df["model"].nunique())

models_we_care = ['A6','Q5','S8']


df["model"] = df["model"].str.strip().str.upper()
models_we_care = [model.strip().upper() for model in models_we_care]
df= df[df["model"].apply(lambda x: x in models_we_care)]


df=df.reset_index()
df=df.drop(columns=["index"])

print(f" \n \n -------------------TEMİZLENMİS HALİ------------------------- ")
print(df)
print(df["model"].unique())
print(df["model"].nunique())
print(df.columns)


#Arabanın fiyatını tahmin etmeye çalışan bir model inşa edeceğim.

label_encoder = LabelEncoder()

df = pd.get_dummies(df,columns=["model"])

vites_ustunlugu = {
    'Manual' : 1,
    'Semi-Auto' : 2,
    'Automatic' : 3
}

df["transmission"] =df["transmission"].replace(vites_ustunlugu)

yakıt_ustunlugu = {
  'Diesel':1,
  'Petrol':2,
  'Hybrid':3
}


df["fuelType"] = df["fuelType"].replace(yakıt_ustunlugu)

df=df.astype(int)

print(f"\n \n \n ------------------------ENCODE EDİLMİS TABLO ----------------------")

print(f"{df.head(30)} \n \n")

yedek = df.copy()
Y = df["price"]
X = df.drop(columns=["price"])
X = X.drop(columns=["tax"])

x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.35,random_state=0)

lr = LinearRegression()

lr.fit(x_train,y_train)

coefficients = lr.coef_
intercept = lr.intercept_

"""""
print(f"katsayılar : {coefficients}")
print(f"intercept  : {intercept}")

"""""


y_predict_train = lr.predict(x_train)
y_predict_test = lr.predict(x_test)

"""""

print("GERCEK DEGERLER :")
print(y_test.values)
print(f" \t \t \t TAHMİN DEGERLER :")
print(y_predict_test)

"""""

gercek_degerler_frame = pd.DataFrame(y_test.values,columns=["Gercek degerler"])
tahmin_degerler_frame = pd.DataFrame(y_predict_test,columns=["Tahmin degerler"])

birlesmis_frame = pd.concat([gercek_degerler_frame,tahmin_degerler_frame],axis=1)

print(f"Lineer regresyon gercek ve tahminleri : \n \n ")
print(birlesmis_frame.head(15))

X_train_sm = sm.add_constant(x_train)  # Eğitim verisi için
X_test_sm = sm.add_constant(x_test)    # Test verisi için

model1 = sm.OLS(y_train, X_train_sm).fit()
print("\n OLS Model Özeti:")
print(model1.summary())

print("\n Lineer Regresyon R2 sonucu")
print(f" \n {r2_score(gercek_degerler_frame.values,tahmin_degerler_frame.values)}")

lr2 = LinearRegression()

lr2.fit(x_train,y_train)

X_train_sm = sm.add_constant(x_train)  # Eğitim verisi için
X_test_sm = sm.add_constant(x_test)    # Test verisi için

model2 = sm.OLS(y_train, X_train_sm).fit()
print("\n OLS Model Özeti:")
print(model2.summary())


#polinomal regresyon ve karar ağacına da bakıp birkaç görsellestirme yapacağım

x_reg = PolynomialFeatures(degree=3)


x_poly_train = x_reg.fit_transform(x_train)


lr3 = LinearRegression()
lr3.fit(x_poly_train, y_train)  # y_test yerine y_train kullanılmalı


x_poly_test = x_reg.transform(x_test)  # fit_transform yerine sadece transform
poly_tahmin_degerler = pd.DataFrame(lr3.predict(x_poly_test), columns=["Tahmin Degerler"])


birlesmis_poly_frame = pd.concat([gercek_degerler_frame, poly_tahmin_degerler], axis=1)


print(f" \n \n Polinomal regresyon icin : {birlesmis_poly_frame.head(15)} \n \n")

print(f"Polinomal Regresyon icin R2 degeri {r2_score(gercek_degerler_frame.values,poly_tahmin_degerler.values)} \n \n ")

rfr_regressor = RandomForestRegressor(random_state=0)

rfr_regressor.fit(x_train,y_train)

rassal_tahminleri = pd.DataFrame(rfr_regressor.predict(x_test),columns=["Rassal Orman Tahminleri"])

rassal_dataframe = pd.concat([rassal_tahminleri,gercek_degerler_frame],axis=1)

print(rassal_dataframe.head(15))

print(f" \n Rassal Orman R2 degeri : {r2_score(gercek_degerler_frame.values,rassal_tahminleri.values)} \n \n")

print(f" \t \t \t \t -------------------------------   T U M    R2   S O N U C L A R I --------------------------------        ")

print("\n Lineer Regresyon icin R2 degeri")
print(f" \n {r2_score(gercek_degerler_frame.values,tahmin_degerler_frame.values)}")

print(" \n Polinomal Regresyon icin R2 degeri : \n ")
print(r2_score(gercek_degerler_frame.values,poly_tahmin_degerler.values))

print(" \n Rassal Ormanlar icin R2 degeri : \n ")
print(r2_score(gercek_degerler_frame.values,rassal_tahminleri.values))
print("\n\n")


years_we_visual = [2012,2013,2014,2015,2016,2017,2018,2019,2020]
X= df[df["year"].apply(lambda x: x in years_we_visual)]


print("Gorsellestirme yapacağımız yıllar")
print(X["year"].unique())
print("\n\n")

X2 = X["year"].values.reshape(-1,1)

Y2 = X["price"]


lr3.fit(X2,Y2)

plt.title("LİNEER REGRESYONLA ARABANIN ÜRETİM YILINA GÖRE ARABA FİYAT DEĞİŞİMİ")
plt.xlabel("ÜRETİM YILI")
plt.ylabel("FİYATLAR")
plt.scatter(X2,Y2,color="red")
plt.plot(X2,lr3.predict(X2),color="blue")
plt.show()

x_reg2 = PolynomialFeatures(degree=4)

x_poly_productdate = x_reg2.fit_transform(X2)

x_reg2.fit(x_poly_productdate,Y2)

lr3.fit(x_poly_productdate,Y2)

plt.title("POLİNOMAL REGRESYONLA ARABANIN ÜRETİM YILINA GÖRE ARABA FİYAT DEĞİŞİMİ")
plt.xlabel("ÜRETİM YILI")
plt.ylabel("FİYATLAR")
plt.scatter(X2,Y2,color="red")
plt.plot(X2,lr3.predict(x_poly_productdate),color="blue")
plt.show()


yedek = yedek[(yedek["mileage"] >= 20000) & (yedek["mileage"] <= 60000)]
yedek = yedek.drop(columns=["tax"])

X3 = yedek["mileage"].values.reshape(-1,1)
Y3 = yedek["price"]


x_poly_year = x_reg2.fit_transform(X3)

lr3.fit(x_poly_year,Y3)

plt.title("SÜRÜLEN KM YE GÖRE ARABA FİYATI DEĞİŞİMİ")
plt.xlabel("SÜRÜLEN KM")
plt.ylabel("FİYATLAR")
plt.scatter(X3,Y3,color="red")
plt.plot(X3,lr3.predict(x_poly_year),color="blue")
plt.show()


print("Grafiklerde ele alınan veri sayıları")
print(f"\n  X :{len(X3)}")
print(f"\n  Y :{len(Y3)}")

print(X)

X=X.drop(columns=["price","tax"])

print("2020 yılında üretilen,yarı-otomatik vitesli, 35.000 km de , benzinli ,57 mpg li , 2.0 motor , A6 arabanın fiyatının modellerle tahmini : ")

car_features = np.array([2020, 2, 35000, 2, 57, 2, 1, 0, 0])


print("\n\n\nLineer regresyonla tahmini fiyat : ")
print(lr.predict([car_features]))
print("\n")

print("\n\nRassal ormanlarla tahmini fiyat : ")
print(rfr_regressor.predict([car_features]))
print("\n\n")










































