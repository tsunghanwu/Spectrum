#%% Simulating raw data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

t_train = np.linspace(0, 300, 3001)
t_test = np.linspace(300, 400, 1001)

Mag = 40000
Freq = 0.015
Offset = 600
Decay = -5
Const = 300000

Noise = 5000

np.random.seed(1)
intensity_train = Mag*np.sin(Freq*t_train + Offset) + Decay*t_train + Noise*np.random.rand(3001) + Const
intensity_test = Mag*np.sin(Freq*t_test + Offset) + Decay*t_test + Noise*np.random.rand(1001) + Const

fig, ax = plt.subplots()
plt.plot(t_train, intensity_train)
plt.plot(t_test, intensity_test)
plt.title("Intensity vs. time")
plt.xlabel("Time")
plt.ylabel("Intensity")

# Building regression model 

fitfunc = lambda p, x: p[0]*np.sin(p[1]*x + p[2]) + p[3]*x + p[4]
errfunc = lambda p, x, y: fitfunc(p,x) - y
p0 = [0.5*(np.max(intensity_train)-np.min(intensity_train)), 0.03, 1, -2, np.mean(intensity_train)]
p1, success = optimize.leastsq(errfunc, p0[:], args=(t_train, intensity_train))
train_predictions = fitfunc(p1, t_train)
test_predictions = fitfunc(p1, t_test)

mae_train = mean_absolute_error(intensity_train, train_predictions)
mse_train = mean_squared_error(intensity_train, train_predictions)
rmse_train = np.sqrt(mse_train)

mae_predict = mean_absolute_error(intensity_test, test_predictions)
mse_predict = mean_squared_error(intensity_test, test_predictions)
rmse_predict = np.sqrt(mse_predict)

period = 2*np.pi/p1[1] 
fringe_count = t_test.max() / period

plt.plot(t_train, train_predictions, label="Train set")
plt.plot(t_test, test_predictions, label="Test set")
plt.legend()
ax.text(250,280000,"MAE_train = {0:.2f}\nMAE_predict = {1:.2f}\nRMSE_train = {2:.2f}\nRMSE_predict = {3:.2f}\nFringe_count = {4:.2f}".format(mae_train,mae_predict,rmse_train,rmse_predict,fringe_count))
plt.show()

#%% Evaluation on noise levels

noise_levels = [i for i in np.linspace(0,10000,num=11)]
mae_list_train = []
mae_list_predict = []
rmse_list_train = []
rmse_list_predict = []
fc_list = []

fig2, ax2 = plt.subplots()
for noise_level in noise_levels:
    np.random.seed(1)
    intensity_train = Mag*np.sin(Freq*t_train + Offset) + Decay*t_train + noise_level*np.random.rand(3001) + Const
    intensity_test = Mag*np.sin(Freq*t_test + Offset) + Decay*t_test + noise_level*np.random.rand(1001) + Const

    fitfunc = lambda p, x: p[0]*np.sin(p[1]*x + p[2]) + p[3]*x + p[4]
    errfunc = lambda p, x, y: fitfunc(p,x) - y
    p0 = [0.5*(np.max(intensity_train)-np.min(intensity_train)), 0.03, 1, -2, np.mean(intensity_train)]
    p1, success = optimize.leastsq(errfunc, p0[:], args=(t_train, intensity_train))
    train_predictions = fitfunc(p1, t_train)
    test_predictions = fitfunc(p1, t_test)
    
    mae_train = mean_absolute_error(intensity_train, train_predictions)
    mae_predict = mean_absolute_error(intensity_test, test_predictions)
    mse_train = mean_squared_error(intensity_train, train_predictions)
    mse_predict = mean_squared_error(intensity_test, test_predictions)
    rmse_train = np.sqrt(mse_train)
    rmse_predict = np.sqrt(mse_predict)
    mae_list_train.append(mae_train)
    mae_list_predict.append(mae_predict)
    rmse_list_train.append(rmse_train)
    rmse_list_predict.append(rmse_predict)
    
    period = 2*np.pi/p1[1] 
    fringe_count = t_test.max() / period
    fc_list.append(fringe_count)
    plt.plot(t_train, train_predictions, label="Train-{0}".format(noise_level))
    plt.plot(t_test, test_predictions, label="Test-{0}".format(noise_level))
    plt.legend()
    plt.show()
    
fig3 = plt.figure()
ax3_1 = fig3.add_subplot(3,1,1)
plt.plot(noise_levels, mae_list_train, "-o", label="MAE_train")
plt.plot(noise_levels, mae_list_predict, "-o", label="MAE_predict")
plt.legend()

ax3_2 = fig3.add_subplot(3,1,2)
plt.plot(noise_levels, rmse_list_train, "-o", label="RMSE_train")
plt.plot(noise_levels, rmse_list_predict, "-o", label="RMSE_predict")
plt.legend()

ax3_3 = fig3.add_subplot(3,1,3)
plt.plot(noise_levels, fc_list, "-o", label="Fringe count")
plt.legend()
plt.xlabel("Noise levels")
plt.show()

#%% Evaluation on training sample size
train_split_ratio= [0.1*i for i in np.linspace(2,9,num=8)]
mae_list_train = []
mae_list_predict = []
rmse_list_train = []
rmse_list_predict = []
fc_list = []

for split in train_split_ratio:
    boundary = np.floor(400*split)
    t_train_split = np.linspace(0,boundary,boundary*10)
    t_test_split = np.linspace(boundary,400,4000-boundary*10)
    
    np.random.seed(1)
    intensity_train_split = Mag*np.sin(Freq*t_train_split + Offset) + Decay*t_train_split + Noise*np.random.rand(int(boundary*10)) + Const
    intensity_test_split = Mag*np.sin(Freq*t_test_split + Offset) + Decay*t_test_split + Noise*np.random.rand(int(4000-boundary*10)) + Const
    fitfunc = lambda p, x: p[0]*np.sin(p[1]*x + p[2]) + p[3]*x + p[4]
    errfunc = lambda p, x, y: fitfunc(p,x) - y
    p0 = [0.5*(np.max(intensity_train_split)-np.min(intensity_train_split)), 0.03, 1, -2, np.mean(intensity_train_split)]
    p1, success = optimize.leastsq(errfunc, p0[:], args=(t_train_split, intensity_train_split))
    train_predictions_split = fitfunc(p1, t_train_split)
    test_predictions_split = fitfunc(p1, t_test_split)
    
    mae_train = mean_absolute_error(intensity_train_split, train_predictions_split)
    mae_predict = mean_absolute_error(intensity_test_split, test_predictions_split)
    mse_train = mean_squared_error(intensity_train_split, train_predictions_split)
    mse_predict = mean_squared_error(intensity_test_split, test_predictions_split)
    rmse_train = np.sqrt(mse_train)
    rmse_predict = np.sqrt(mse_predict)
    mae_list_train.append(mae_train)
    mae_list_predict.append(mae_predict)
    rmse_list_train.append(rmse_train)
    rmse_list_predict.append(rmse_predict)
    
    period = 2*np.pi/p1[1] 
    fringe_count = t_test_split.max() / period
    fc_list.append(fringe_count)
    
fig4 = plt.figure()
ax4_1 = fig4.add_subplot(1,3,1)
plt.plot(train_split_ratio, mae_list_train, "-o", label="MAE_train")
plt.plot(train_split_ratio, mae_list_predict, "-o", label="MAE_predict")
plt.xlabel("Train set split ratio")
plt.legend()

ax4_2 = fig4.add_subplot(1,3,2)
plt.plot(train_split_ratio, rmse_list_train, "-o", label="RMSE_train")
plt.plot(train_split_ratio, rmse_list_predict, "-o", label="RMSE_predict")
plt.xlabel("Train set split ratio")
plt.legend()

ax4_3 = fig4.add_subplot(1,3,3)
plt.plot(train_split_ratio, fc_list, "-o", label="Fringe count")
plt.legend()
plt.xlabel("Train set split ratio")
plt.show()

#%% Testing on actual dataset

data = pd.read_csv("N9GM12_16 LSRi 550nm.csv", header=2)
data = data.iloc[:,1:3]
df = pd.DataFrame()
df["Time"] = data.iloc[:,0]
df["Intensity"] = data.iloc[:,1]

train = df.iloc[0:2500]
test = df.iloc[2500:]

fig5, ax5 = plt.subplots()
plt.plot(df["Time"], df["Intensity"])

fitfunc = lambda p, x: p[0]*np.sin(p[1]*x + p[2]) + p[3]*x + p[4]
errfunc = lambda p, x, y: fitfunc(p,x) - y

p0_actual = [0.5*(np.max(train["Intensity"])-np.min(train["Intensity"])), 0.03, 1, -2, np.mean(train["Intensity"])]
p1_actual, success = optimize.leastsq(errfunc, p0_actual[:], args=(train["Time"], train["Intensity"]))

train_predictions2 = fitfunc(p1_actual, train["Time"])
test_predictions2 = fitfunc(p1_actual, test["Time"])

mae2_train = mean_absolute_error(train["Intensity"], train_predictions2)
mae2_predict = mean_absolute_error(test["Intensity"], test_predictions2)
mse2_train = mean_absolute_error(train["Intensity"], train_predictions2)
mse2_predict = mean_squared_error(test["Intensity"], test_predictions2)
rmse2_train = np.sqrt(mse2_train)
rmse2_predict = np.sqrt(mse2_predict)

period2 = 2*np.pi/p1_actual[1] 
fringe_count2 = test["Time"].max() / period2

plt.plot(train["Time"], train_predictions2, label="Train set")
plt.plot(test["Time"], test_predictions2, label="Test set")
plt.title("Intensity vs. time")
plt.xlabel("Time")
plt.ylabel("Intensity")
plt.legend()
ax5.text(250,300000,"MAE_train = {0:.2f}\nMAE_predict = {1:.2f}\nRMSE_train = {2:.2f}\nRMSE_predict = {3:.2f}\nFringe_count = {4:.2f}".format(mae2_train,mae2_predict,rmse2_train,rmse2_predict,fringe_count2))
plt.show()

#%% Evaluating training set split ratio on actual data

train_split_ratio = [0.1*i for i in np.linspace(2,9,num=8)]
mae2_list_train = []
mae2_list_predict = []
rmse2_list_train = []
rmse2_list_predict = []
fc2_list = []

for split in train_split_ratio:
    boundary = int(np.floor(df["Time"].max()*split))
    train = df.iloc[0:boundary*10]
    test = df.iloc[boundary*10:]
    
    fitfunc = lambda p, x: p[0]*np.sin(p[1]*x + p[2]) + p[3]*x + p[4]
    errfunc = lambda p, x, y: fitfunc(p,x) - y
    
    p0_actual = [0.5*(np.max(train["Intensity"])-np.min(train["Intensity"])), 0.03, 1, -2, np.mean(train["Intensity"])]
    p1_actual, success = optimize.leastsq(errfunc, p0_actual[:], args=(train["Time"], train["Intensity"]))
    
    train_predictions2 = fitfunc(p1_actual, train["Time"])
    test_predictions2 = fitfunc(p1_actual, test["Time"])
    
    mae2_train = mean_absolute_error(train["Intensity"], train_predictions2)
    mae2_predict = mean_absolute_error(test["Intensity"], test_predictions2)
    mse2_train = mean_absolute_error(train["Intensity"], train_predictions2)
    mse2_predict = mean_squared_error(test["Intensity"], test_predictions2)
    rmse2_train = np.sqrt(mse2_train)
    rmse2_predict = np.sqrt(mse2_predict)
   
    mae2_list_train.append(mae2_train)
    mae2_list_predict.append(mae2_predict)
    rmse2_list_train.append(rmse2_train)
    rmse2_list_predict.append(rmse2_predict)
    
    period2 = 2*np.pi/p1_actual[1] 
    fringe_count2 = test["Time"].max() / period2
    fc2_list.append(fringe_count2)
    
fig6 = plt.figure()
ax6_1 = fig6.add_subplot(1,3,1)
plt.plot(train_split_ratio, mae2_list_train, "-o", label="MAE_train")
plt.plot(train_split_ratio, mae2_list_predict, "-o", label="MAE_predict")
plt.xlabel("Train set split ratio")
plt.legend()

ax6_2 = fig6.add_subplot(1,3,2)
plt.plot(train_split_ratio, rmse2_list_train, "-o", label="RMSE_train")
plt.plot(train_split_ratio, rmse2_list_predict, "-o", label="RMSE_predict")
plt.xlabel("Train set split ratio")
plt.legend()

ax6_3 = fig6.add_subplot(1,3,3)
plt.plot(train_split_ratio, fc2_list, "-o", label="Fringe count")
plt.legend()
plt.xlabel("Train set split ratio")
plt.show()

