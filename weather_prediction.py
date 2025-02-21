import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import statsmodels.api as sm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Veri yükleme
weather_prediction = pd.read_csv("weather_data.csv")
df = pd.DataFrame(weather_prediction)

# windy ve play sütunlarını sayısal hale getirme
label_encoder = LabelEncoder()
df["windy"] = label_encoder.fit_transform(df["windy"])
df["play"] = label_encoder.fit_transform(df["play"])

# outlook sütununu dummies haline getirme
df = pd.get_dummies(df, columns=["outlook"])
df=df.astype(int)

print(df.columns)

# Bağımlı ve bağımsız değişkenleri ayırma
y1 = df["temperature"]
df.drop(columns=["temperature"], inplace=True)
x1 = df

print(df.columns)


# Veriyi train/test olarak ayırma
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.33, random_state=0)

"""  

Veriyi standartlaştırma
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

"""""

# Modeli eğitme
rl = LinearRegression()
rl.fit(x_train, y_train)

# Tahmin yapma
y_pred = rl.predict(x_train)

# Tahmin ve gerçek değerleri gösterme
print(df)
print("Tahmin Edilen Değerler:", y_pred)
print("Gerçek Değerler:", y_test.values)

print(f"Gerçek min - max degerleri : {y_test.min()} ve {y_test.max()}")
print(f"Tahminin min ve max degerleri : {y_pred.min()} ve {y_pred.max()}")

# Statsmodels kullanarak OLS regresyonu
X = np.append(arr = np.ones((len(df),1)).astype(int), values=df, axis=1)

# Tüm değişkenleri kullanarak OLS
X_l = df.values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(y1, X_l).fit()
print(model.summary())


df.drop(columns=["play"], inplace=True)

rl.fit(x_train,y_train)

y_predf = rl.predict(x_test)

print("Once Tahmin Edilen Degerler : ",y_pred)
print("Sonra Tahmin Edilen Değerler:", y_predf)
print("Gerçek Değerler:", y_test.values)







