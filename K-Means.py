import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#kmeans bir kümeleme algoritmasıdır

df = pd.read_csv("social-media.csv")

print(df)

yas = df["Yas"]

hacim = df["Hacim"]

yas_hacim = pd.concat([df["Yas"],df["Hacim"]],axis=1)

print(yas_hacim)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3 , init='k-means++')

kmeans.fit(yas_hacim)

print(kmeans.cluster_centers_)

liste = []

for i in range(1,11):

    kmeans2= KMeans(n_clusters=i , init='k-means++')
    kmeans2.fit(yas_hacim)
    print(f"i = {i} için k-means {kmeans2.cluster_centers_} \n")
    liste.append(kmeans2.inertia_)

plt.plot(range(1,11),liste)

plt.show()

""""
grafikteki dirsek noktası önemlidir , bu bize en uygun küme sayısının kaç olduğunu göstermede yardımcı olur.

"""""





