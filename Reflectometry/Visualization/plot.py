import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.read_csv("Spectrum.txt", sep=" ", skiprows=9)
df = df.iloc[:,:-1]

# Rearranging column names
columns = list(df.columns)
columns = [float(i) for i in columns]
columns = [np.round(i) for i in columns]
df.columns = columns

# Set time as index
time = [float(10*(i)) for i in range(df.shape[0])]
df.index = time

# Create heatmap visualization
fig, ax = plt.subplots()
sns.heatmap(df, cmap="gnuplot")
plt.title("Spectrum")
plt.xlabel("Wavelength")
plt.ylabel("Time")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Create 3D contour visualization
x, y = np.meshgrid(df.columns, df.index)
fig2 = plt.figure()
ax = fig2.gca(projection='3d')
surf = ax.plot_surface(x, y, df, cmap="gnuplot")
ax.set_title("3D Surface Plot")
ax.set_xlabel("Wavelength")
ax.set_ylabel("Time")
ax.set_zlabel("Intensity")
plt.show()

# Normalization
scaler = MinMaxScaler()
df_T = df.transpose()
df_T_norm = scaler.fit_transform(df_T)
df_norm = df_T_norm.transpose()
df_norm = pd.DataFrame(data=df_norm, index=df.index, columns=df.columns)

# Create heatmap visualization
fig3, ax = plt.subplots()
sns.heatmap(df_norm, cmap="gnuplot")
plt.title("Normalized Spectrum")
plt.xlabel("Wavelength")
plt.ylabel("Time")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()
          
# Create 3D contour visualization
x, y = np.meshgrid(df_norm.columns, df_norm.index)
fig4 = plt.figure()
ax = fig4.gca(projection='3d')
surf = ax.plot_surface(x, y, df_norm, cmap="gnuplot")
ax.set_title("3D Surface Plot")
ax.set_xlabel("Wavelength")
ax.set_ylabel("Time")
ax.set_zlabel("Intensity")
plt.show()