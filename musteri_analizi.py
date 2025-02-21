import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.txt")

pd.set_option("display.max_rows", None)  # Satır sınırını kaldır
pd.set_option("display.max_columns", None)  # Sütun sınırını kaldır

print(data.columns)

target = data["Exited"] #Y
data = data.drop(columns=["RowNumber","CustomerId","Surname","Exited"]) #X

print(data.head())
print(data.columns)
print(f"Toplam farklı ülkeler : {data['Geography'].nunique()}")

data = pd.get_dummies(data,drop_first=True).astype(int) #true false yerine 0 ve 1 yazıyoruz astype sayesinde

print("--------------------------- D A T A --- S O N --------- H A L İ ----------------")
print(data.head())

#kategorik verilerimizi de sayısal hale dönüştürdüğümüze göre artık işlemlere başlayabiliriz.

x_train , x_test , y_train , y_test = train_test_split(data,target,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

X_train = StandardScaler.fit_transform(x_train)
X_test = StandardScaler.fit_transform(x_test)

import keras
from keras import Sequential
from keras import Dense

classifier = Sequential()

classifier.add(Dense(6,init="uniform",activation='relu',input_dim=11))
classifier.add(Dense(6,init="uniform",activation='relu'))
classifier.add(Dense(1,init="uniform",activation="sigmoid"))

classifier.compile(optimizer=keras.optimizers.Adam,loss=keras.losses.binary_crossentropy,metrics=keras.metrics.Accuracy)

classifier.fit(X_train,y_train,epochs=100,batch_size=64,shuffle=True)

y_pred = classifier.predict(X_test)



    

