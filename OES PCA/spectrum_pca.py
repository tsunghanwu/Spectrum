import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

df = pd.read_excel("Cleaned_df.xlsx", index_col=0)

scaler = MinMaxScaler()
df_norm = scaler.fit_transform(df.transpose())
df_norm = df_norm.transpose()

# Nomalized data
pca1_3 = PCA(n_components=3)
pca1_2 = PCA(n_components=2)
df_reduced_3 = pca1_3.fit_transform(df_norm)
df_reduced_2 = pca1_2.fit_transform(df_norm)
comp1 = pca1_3.components_
exp_var1 = pca1_3.explained_variance_ratio_

# Unnormalized data
pca2_3 = PCA(n_components=3)
pca2_2 = PCA(n_components=2)
df2_reduced_3 = pca2_3.fit_transform(df)
df2_reduced_2 = pca2_2.fit_transform(df)
comp2 = pca2_3.components_
exp_var2 = pca2_3.explained_variance_ratio_

# Plotting PC1, PC2, PC3 for normalized data
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df_reduced_3[:,0], df_reduced_3[:,1], df_reduced_3[:,2], label="Normalized")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend()
plt.tight_layout()
plt.show()

# Plotting PC1, PC2, PC3 for unnormalized data
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection="3d")
ax.scatter(df2_reduced_3[:,0], df2_reduced_3[:,1], df2_reduced_3[:,2], label="Unnormalized", c="orange")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend()
plt.tight_layout()
plt.show()

# Plotting PC1, PC2 for normalized data
fig2 = plt.figure()
ax = fig2.add_subplot(111)
plt.scatter(df_reduced_2[:,0], df_reduced_2[:,1], label="Normalized")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.show()

# Plotting PC1, PC2 for unnormalized data
fig3 = plt.figure()
ax = fig3.add_subplot(111)
plt.scatter(df2_reduced_2[:,0], df2_reduced_2[:,1], label="Unnormalized", c="orange")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.show()

# Plotting unnormalized OES spectrum
fig4 = plt.figure()
ax1 = fig4.add_subplot(411)
plt.plot(df.columns, df.loc["Step_A"], c="orange")
plt.title("OES Spectrum - Step_A")
ax2 = fig4.add_subplot(412)
plt.plot(df.columns, df.loc["Step_B"], c="orange")
plt.title("OES Spectrum - Step_B")
ax3 = fig4.add_subplot(413)
plt.plot(df.columns, df.loc["Step_C"], c="orange")
plt.title("OES Spectrum - Step_C")
ax4 = fig4.add_subplot(414)
plt.plot(df.columns, df.loc["Step_D"], c="orange")
plt.title("OES Spectrum - Step_D")
plt.tight_layout()
plt.show()

# Plotting loading for unnormalized OES spectrum
fig5 = plt.figure()
ax1 = fig5.add_subplot(411)
plt.plot(df.columns, comp1[0,:], c="orange")
plt.title("PC1 Loading - Unnormalized")
ax2 = fig5.add_subplot(412)
plt.plot(df.columns, comp1[1,:], c="orange")
plt.title("PC2 Loading - Unnormalized")
ax3 = fig5.add_subplot(413)
plt.plot(df.columns, comp1[2,:], c="orange")
plt.title("PC3 Loading - Unnormalized")
ax3 = fig5.add_subplot(414)
plt.bar(["PC1", "PC2", "PC3"], exp_var1, color="orange")
plt.title("Explained Variance Ratio - Normalized")
plt.tight_layout()
plt.show()

# Plotting normalized OES spectrum
fig6 = plt.figure()
ax1 = fig6.add_subplot(411)
plt.plot(df.columns, df_norm[0,:])
plt.title("OES Normalized Spectrum - Step_A")
ax2 = fig6.add_subplot(412)
plt.plot(df.columns, df_norm[1,:])
plt.title("OES Normalized Spectrum - Step_B")
ax3 = fig6.add_subplot(413)
plt.plot(df.columns, df_norm[2,:])
plt.title("OES Normalized Spectrum - Step_C")
ax4 = fig6.add_subplot(414)
plt.plot(df.columns, df_norm[3,:])
plt.title("OES Normalized Spectrum - Step_D")
plt.tight_layout()
plt.show()

# Plotting loading for normalized OES spectrum
fig7 = plt.figure()
ax1 = fig7.add_subplot(411)
plt.plot(df.columns, comp1[0,:])
plt.title("PC1 Loading - Normalized")
ax2 = fig7.add_subplot(412)
plt.plot(df.columns, comp1[1,:])
plt.title("PC2 Loading - Normalized")
ax3 = fig7.add_subplot(413)
plt.plot(df.columns, comp1[2,:])
plt.title("PC3 Loading - Normalized")
ax3 = fig7.add_subplot(414)
plt.bar(["PC1", "PC2", "PC3"], exp_var1)
plt.title("Explained Variance Ratio - Normalized")
plt.tight_layout()
plt.show()