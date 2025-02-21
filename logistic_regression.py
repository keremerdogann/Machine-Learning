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
from sklearn.linear_model import LogisticRegression


#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)


df = pd.read_excel("sadi_hoca_veri_seti.xlsx")

print(df)

Y = df["cinsiyet"]
X = df.drop(columns=["cinsiyet","ulke"])

x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.35,random_state=0)

sc=StandardScaler()

sc_x_train = sc.fit_transform(x_train)
sc_x_test = sc.transform(x_test)

log_r = LogisticRegression(random_state=0)

log_r.fit(sc_x_train,y_train)

y_pred = log_r.predict(sc_x_test)
y_real = y_test

print(" \n Tahmin degerleri : \n")
print(y_pred)

print(" \n Gercek Degerler : \n")
print(y_real.values)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_real,y_pred)

print(cm)







