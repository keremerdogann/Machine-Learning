import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.max_rows",None)
pd.set_option("display.min_rows",None)
pd.set_option('display.max_colwidth', None)


df = pd.read_csv('Cost_of_Living_Index_by_Country_2024.csv')

print(df.columns)


### KOLONLARDA TÜRKÇELEŞTİRME YAPTIM ###

df["Yerel Satın Alma Gücü"] = df["Local Purchasing Power Index"]
df["Yaşam Maliyeti Endeksi"] = df["Cost of Living Index"]
df["Market Endeksi"] = df["Groceries Index"]
df["Restoran Fiyat Endeksi"] = df["Restaurant Price Index"]
df["Kira Endeksi"]=df["Rent Index"]

df.drop(["Local Purchasing Power Index", "Cost of Living Index", "Groceries Index", "Restaurant Price Index","Rent Index"], axis=1, inplace=True)

### TÜRKÇELEŞTİRME KISMI BİTİŞ ###


top_5_countries = df.nlargest(7, 'Market Endeksi')[['Market Endeksi']].reset_index()


top_5_countries['Country'] = df.nlargest(7, 'Market Endeksi')['Country']

all_countries=df[["Country","Yaşam Maliyeti Endeksi"]]


plt.figure(figsize=(12, 8))
sns.barplot(x='Country', y='Market Endeksi', data=top_5_countries, palette='viridis')
plt.title('En Yüksek Market Endeksine Sahip 5 Ülke', fontsize=20)
plt.xlabel('Ülke', fontsize=14)
plt.ylabel('Market Endeksi', fontsize=14)
plt.xticks(rotation=45)
plt.show()



# Sütun grafiği oluşturma
plt.figure(figsize=(12, 8))
sns.barplot(x='Country', y='Yaşam Maliyeti Endeksi', data=all_countries, palette='viridis')
plt.title('ÜLKE VE MARKET ENDEKSİ ARASINDAKİ İLİŞKİ', fontsize=20)
plt.xlabel('Ülke', fontsize=14)
plt.ylabel('Market Endeksi', fontsize=14)
plt.xticks(rotation=45)
plt.show()


ranks=df[["Rank","Yaşam Maliyeti Endeksi"]]


plt.figure(figsize=(12, 8))
sns.scatterplot(x='Rank', y='Yaşam Maliyeti Endeksi', data=ranks)
plt.title('Rank ve Yaşam Maliyeti Arasındaki ilişki', fontsize=20)
plt.xlabel('Rank', fontsize=14)
plt.ylabel('Yaşam Maliyeti Endeksi', fontsize=14)
plt.xticks(rotation=45)
plt.show()







plt.figure(figsize=(10, 6))
sns.scatterplot(x='Yerel Satın Alma Gücü', y='Yaşam Maliyeti Endeksi', data=df)
plt.title('Yerel Satın Alma Gücü ve Yaşam Maliyeti İlişkisi', fontsize=30)
plt.xlabel('Yerel Satın Alma Gücü Endeksi', fontsize=14)
plt.ylabel('Yaşam Maliyeti Endeksi', fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Market Endeksi', y='Restoran Fiyat Endeksi', data=df)
plt.title('Market Endeksi ve Restoran Fiyat Endeksi İlişkisi', fontsize=30)
plt.xlabel('Market Endeksi', fontsize=14)
plt.ylabel('Restoran Fiyat Endeksi', fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Yerel Satın Alma Gücü', y='Market Endeksi', data=df)
plt.title('Yerel Satın Alma Gücü ve Market Endeksi', fontsize=30)
plt.xlabel('Yerel Satın Alma Gücü', fontsize=14)
plt.ylabel('Market Endeksi', fontsize=14)
plt.show()


plt.figure(figsize=(15,9))
sns.scatterplot(x="Kira Endeksi",y="Market Endeksi",data=df)
plt.title("KİRA VE MARKET FİYATLARI BİRLİKTE ARTIŞ GÖSTERİYOR MU ?")
plt.xlabel("KİRA ENDEKSİ")
plt.ylabel("MARKET ENDEKSİ")
#plt.show()


print(df.columns)
print()



df_sorted = df[['Country', 'Yaşam Maliyeti Endeksi']].sort_values(by='Yaşam Maliyeti Endeksi', ascending=False).reset_index(drop=True)


plt.figure(figsize=(15, 10))
sns.barplot(x='Yaşam Maliyeti Endeksi', y='Country', data=df_sorted, palette='coolwarm')
plt.title('Ülkelerin Yaşam Maliyeti Endeksine Göre Sıralanması', fontsize=20)
plt.xlabel('Yaşam Maliyeti Endeksi', fontsize=14)
plt.ylabel('Ülke', fontsize=10)
plt.show()

