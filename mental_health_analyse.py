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

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)


df = pd.read_csv("mental_health_and_technology_usage_2024.csv")
#print(df)

print(df["Stress_Level"].unique())
print(df["Mental_Health_Status"].unique())


level_of_stress = {
    'Low' : 1,
    'Medium' : 2,
    'High' : 3
}

Y = df["Stress_Level"].replace(level_of_stress)
X = df.drop(columns=["Stress_Level","Work_Environment_Impact","Online_Support_Usage"])


label_encoder = LabelEncoder()

X["Gender"] = label_encoder.fit_transform(X["Gender"])

print("Replace hale gelmiş stres seviyesi")
print(Y)

X["Support_Systems_Access"] = label_encoder.fit_transform(X["Support_Systems_Access"])

mental_health_status = {

    'Poor' : 1,
    'Fair' : 2,
    'Good' : 3,
    'Excellent' : 4
}


X["Mental_Health_Status"]= X["Mental_Health_Status"].replace(mental_health_status)

X = X.set_index("User_ID")

print(X)

x_activity = df["Physical_Activity_Hours"]

lr = LinearRegression()

lr.fit(x_activity,Y)

y_pred = lr.predict(x_activity)

print("\nTahmini deger : ")
print(y_pred)

print("\nGercek deger : ")
print(Y)

plt.scatter(x_activity,Y)
plt.plot(x_activity,lr.predict(x_activity),color="red")
plt.show()




