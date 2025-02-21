#SADİ HOCANIN EGİTİM VİDEOSUNUN DOĞAL DİL İŞLEME MODELİ KISMININ İLK VİDEOSUNDA
#NLP İLE İLGİLİ KAYNAKLARIN OLDUĞU VİDEO VAR. BİRİNCİ VİDEO O DA .

import numpy as np
import pandas as pd
import re

# stem kütüphanesi bize kelimeleri eklerinden ayırarak sadece köklerini incelememizi sağlar.

from nltk.stem.porter import PorterStemmer

just_port = PorterStemmer()


comments = pd.read_csv("spotify_nlp.csv")

comments=comments.dropna()

comments=comments.reset_index(drop=True)

print(comments.head())

#Negatifleri 0 , Pozitifleri 1 yaptık.
comments["label"] = comments["label"].apply(lambda x : 0 if x == "NEGATIVE" else 1)

print(comments.head())

#a dan z ye büyük ve küçük tüm harfler hariç diğer noktalama işaretleri vs yi boşluk yaptık.
#[a-zA-Z] yazsaydık a dan z ye büyük ve küçük tüm harfleri boşluk yapardı
comm = re.sub('[^a-zA-Z]',' ',comments["Review"][5])

print("\n ilk hali :")
print(comments["Review"][5])
print("\n sonraki hali :")
print(comm)

#tüm harfleri küçültme
comm = comm.lower()
print(comm)

comm = comm.split()

stopwords = pd.read_csv("stopwords.csv")

#for kısmı stopwordsleri tespit edip eğer kelime stopword ise onu almıyor , stopword olmayanları alıyor
#just port stem kısmı da for dongusunden gelen kelimelerin koklerini alıyor
comm = [just_port.stem(kelime) for kelime in comm if not kelime in stopwords.values]

print(comm)

#kelimeler liste halindeydi normalde , şimdi aralarına boşluk koyarak string hale getiriyoruz.
#join , splitin tersi gibi düşünülebilir.
comm = ' '.join(comm)

print(comm)

print("HEPSİNİ BİRLEŞTİREREK TEK HALDE YAZALIM :")

derleme = []

for i in range(100):
    comm = re.sub('[^a-zA-Z]', ' ', comments["Review"][i])
    comm = comm.lower()
    comm = comm.split()
    comm = [just_port.stem(kelime) for kelime in comm if not kelime in stopwords.values]
    comm = ' '.join(comm)
    derleme.append(comm)

print(derleme)








