import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("XXXXX.txt", sep=" ", skiprows=9)
df = df.iloc[:, :-1]
col_list = [float(col) for col in df.columns]
time_list = [time/10 for time in df.index]
df.columns = col_list
df.index = time_list

# Unnormlized pca
pca = PCA(n_components=3)
df_trans = pca.fit_transform(df)
df_trans = pd.DataFrame(df_trans)
df_trans.columns = ["PC1", "PC2", "PC3"]
df_trans.index = time_list
loading = pd.DataFrame(pca.components_).transpose()
loading.columns = ["PC1", "PC2", "PC3"]
loading.index = col_list

fig = plt.figure()
ax1_1 = fig.add_subplot(231)
plt.plot(df_trans["PC1"])
plt.title("PC1")
plt.ylabel("Score")
plt.xlabel("Time")

ax1_2 = fig.add_subplot(232)
plt.plot(df_trans["PC2"])
plt.title("PC2")
plt.ylabel("Score")
plt.xlabel("Time")

ax1_3 = fig.add_subplot(233)
plt.plot(df_trans["PC3"])
plt.title("PC3")
plt.ylabel("Score")
plt.xlabel("Time")

ax2_1 = fig.add_subplot(234)
plt.plot(loading["PC1"])
plt.title("Loading")
plt.ylabel("Counts")
plt.xlabel("Wavelength")

ax2_2 = fig.add_subplot(235)
plt.plot(loading["PC2"])
plt.title("Loading")
plt.ylabel("Counts")
plt.xlabel("Wavelength")

ax2_3 = fig.add_subplot(236)
plt.plot(loading["PC3"])
plt.title("Loading")
plt.ylabel("Counts")
plt.xlabel("Wavelength")
plt.show()
