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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

dtc = DecisionTreeClassifier(criterion='entropy')

df = pd.read_excel("sadi_hoca_veri_seti.xlsx")

le=LabelEncoder()

Y = df["cinsiyet"]
X = df.drop(columns=["cinsiyet","ulke"])

Y = le.fit_transform(df["cinsiyet"])

print("erkek kadın sirasi")
print(le.classes_)

x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.35,random_state=0)

dtc.fit(x_train,y_train)

y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("DTC")
print(cm)

rtc = RandomForestClassifier(n_estimators=10,criterion='entropy')

rtc.fit(x_train,y_train)

y_pred = rtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("RTC")
print(cm)

iris = load_iris()

X = iris.data

Y = iris.target

print(Y)







