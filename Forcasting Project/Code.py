import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
import math
import sys
import numpy.random as random
import pylab
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AR
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from matplotlib.pylab import rcParams
from pandas import Series
rcParams['figure.figsize'] = 15, 6

data = pd.read_csv('sshekha4.csv', header=None)
plt.plot(data)

# #Task 1

#Plot of the data shows Variance
#Shifting the data by (minimum value + 1000) since it contains negative entries. Then taking log transform
dataMin = data[0].min()
#print("dataMin: ", dataMin)
#Function to Shift all the values by dataMin
def shiftData(data, dataMin):
    shiftedData = list()
    for i in range(len(data)):
        shiftedData.append(data[i] + dataMin)
    return Series(shiftedData)
shiftedData = shiftData(data[0], ((-1)*dataMin)+1000)
#print(shiftedData)
#plt.plot(shiftedData)
def logTransform(data):
    return math.log10(data)
logTransformedData = shiftedData.apply(logTransform)
cTransformedData = logTransformedData
print(cTransformedData)
plt.plot(cTransformedData)
ax = plt.gca()
ax.set_ylim([2.9, 3.1])

# Experimental Section starts

# After taking log transform, still there is some change in variance in the data. 
# Now, taking the cube root of the log transformed data
# def cuberoot(data):
#     return data**(1/3)
# cTransformedData = logTransformedData.apply(cuberoot)
#print(cTransformedData)
#plt.plot(cTransformedData)

#Experimental Section ends

#Generating training and testing data
trainData = list()
for i in range(1500):
    trainData.append(cTransformedData[i])
train = Series(trainData)
testData = list()
for i in range(1500, len(cTransformedData)):
    testData.append(cTransformedData[i])
test = Series(testData)

# Task 2
def smaRmseCalculator(train, sma_k, k):
    rmse = 0.0
    for i in range(k, len(train)):
        rmse += ((train[i]-sma_k[i])**2)  
#     for i in range(len(train)):
#         print("train[", i, "]: ", train[i])
#         print("sma_k[", i, "]: ", sma_k[i])
    rmse = (rmse/(len(train)-k))**(1/2)
    return rmse
def smaPredictedCalculator(train, k):
    smaPredicted = list()
    for i in range(k):
        smaPredicted.append(0)
    for i in range(k, len(train)):
        add = 0.0
        for j in range(i-k,i):
            add += train[j]
        smaPredicted.append(add/k)
    return Series(smaPredicted)
# Calculating Simple Moving Average by varying k from 2 to 10
sma_rmse_min = sys.float_info.max
sma_rmse_k = 0 # Any arbitrary initial value
sma_rmse_list = list()
sma_k_list = list()
for i in range(2,101):
    sma_i = smaPredictedCalculator(train, i)
    sma_rmse_i = smaRmseCalculator(train, sma_i, i)
    sma_k_list.append(i)
    sma_rmse_list.append(sma_rmse_i)
#     print("Value of k: ", i)
#     print("Value of RMSE: ", sma_rmse_i)
    if(sma_rmse_i < sma_rmse_min):
        sma_rmse_min = sma_rmse_i
        sma_rmse_k = i
        sma_original_list = list()
        sma_predicted_list = list()
        for j in range(i,len(train)):
            sma_original_list.append(train[j])
            sma_predicted_list.append(sma_i[j])
# print(sma_rmse_k)
print("Train Error:", sma_rmse_min)
# print(len(sma_original_list))
plt.plot(sma_k_list, sma_rmse_list)
plt.xlabel('k')
plt.ylabel('RMSE')
plt.plot(sma_predicted_list, label="Predicted Values")
plt.plot(sma_original_list, label="Original Values")
plt.legend()
plt.show()

###  For Test Data starts
sma_test_rmse = smaRmseCalculator(test, smaPredictedCalculator(test, 2), 2)
print("Test RMSE:", sma_test_rmse)
###  For Test Data ends

#Task 3    
def emaRmseCalculator(train, ema_a):
    rmse = 0.0
    for i in range(len(train)):
        rmse += ((train[i]-ema_a[i])**2)    
#     for i in range(len(train)):
#         print("train[", i, "]: ", train[i])
#         print("ema_a[", i, "]: ", ema_a[i])
    rmse = (rmse/len(train))**(1/2)
    return rmse
def emaPredictedCalculator(train, a):
    emaPredicted = list()
    for i in range(len(train)):
        if(i == 0):
            emaPredicted.append(0.0)
        else:
            emaPredicted.append((a*train[i-1]) + ((1-a)*emaPredicted[i-1]))   
    return Series(emaPredicted)
        
ema_rmse_min = sys.float_info.max
ema_rmse_k = 0 # Any arbitrary initial value
ema_a_list = list()
ema_rmse_list = list()
#Calculating Exponential Moving Average by varying a from 0.1 to 0.9
for i in range(9):
    ema_i = emaPredictedCalculator(train, (i+1)/10)
    ema_rmse_i = emaRmseCalculator(train, ema_i)
    ema_a_list.append((i+1)/10)
    ema_rmse_list.append(ema_rmse_i)
#     print("For a =", (i+1)/10, ", RMSE: ", ema_rmse_i)
    if(ema_rmse_i < ema_rmse_min):
        ema_rmse_min = ema_rmse_i
        ema_rmse_k = (i+1)/10
        ema_original_list = list()
        ema_predicted_list = list()
        for j in range(1, len(train)):
            ema_original_list.append(train[j])
            ema_predicted_list.append(ema_i[j])
# print(ema_rmse_k)
print("Train RMSE:", ema_rmse_min)
# plt.plot(ema_a_list, ema_rmse_list)
# plt.xlabel('a')
# plt.ylabel('RMSE')
# print("List:")
# for i in range(len(train)-1):
#     print(ema_original_list[i], " ", ema_predicted_list[i])
plt.plot(ema_original_list, label="Original Values")
plt.plot(ema_predicted_list, label="Predicted Values")
plt.legend()
ax = plt.gca()
ax.set_xlim([200,250])
ax.set_ylim([3.00001, 3.01])
plt.show()

##  For Test Data starts
#Taking a=0.9 as set by the train data because of least RMSE
ema_test_rmse = emaRmseCalculator(test, emaPredictedCalculator(test, 0.9))
print("Test RMSE:", ema_test_rmse)
##  For Test Data ends
    
#Task 4
# Plotting PACF in order to determine the order p of the AR model
plot_pacf(train, lags=5)
plt.xlabel("k")
plt.ylabel("PACF")
#Estimate parameters of AR(p) model
model = AR(train)
model_fit = model.fit(maxlag=2)
print('Coefficients: %s' % model_fit.params)
# plt.plot(train, color='blue')
# plt.plot(model_fit.fittedvalues, color='red')
# # print(len(train))
# # print(len(model_fit.fittedvalues))

#for test data starts
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
# for i in range(len(predictions)):
#     print(predictions[i])
error = mean_squared_error(test, predictions)
ar_rmse = error**(1/2)
print("RMSE (Test Error): ", ar_rmse)
#for test data ends

train_sub = list()
for i in range(2, len(train)):
    train_sub.append(train[i])
train_subseries = Series(train_sub)
plt.plot(train_subseries, color='blue')
plt.plot(model_fit.fittedvalues, color='red')
# plt.plot(model_fit.fittedvalues, label="Predicted Values")
# plt.plot(train_sub, label="Original Values")
plt.legend()
ax = plt.gca()
ax.set_xlim([200,250])
ax.set_ylim([2.999, 3.01])
plt.show()
# # plt.scatter(model_fit.fittedvalues, train_sub)
# # plt.xlabel("predicted values")
# # plt.ylabel("original values")
# # ax = plt.gca()
# # ax.set_xlim([3.0000, 3.0090])
# # ax.set_ylim([2.9995, 3.0095])
print("Fitted Values Length:", len(model_fit.fittedvalues))
error = mean_squared_error(train_subseries, model_fit.fittedvalues)
ar_rmse = error**(1/2)
print("RMSE (Train Error):", ar_rmse)
# # print(model_fit.resid)
# #print(np.std(model_fit.resid))
# #measurements=np.random.normal(0,np.std(model_fit.resid),len(model_fit.resid))
stats.probplot(model_fit.resid,dist="norm", plot=pylab)
plt.hist(model_fit.resid, color='#0504aa')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
print(np.std(model_fit.resid))
plt.scatter(model_fit.fittedvalues, model_fit.resid)
ax = plt.gca()
ax.set_xlim([3.000, 3.009])
ax.set_ylim([-0.0017, 0.0015])

print(len(model_fit.resid))
print(np.mean(model_fit.resid))
print(np.std(model_fit.resid))
# Above printed mean is approximately equal to zero
#Chi Square Test starts
res_mean = 0; res_sd = np.std(model_fit.resid)
bins = [-float('inf')]
for i in range(9):
    p = (i+1)/10. ; z = norm.ppf(p) ; bound = res_mean + (z*res_sd);
#     print(p, '{:6.4f}'.format(z), '{:6.4f}'.format(bound))
    bins.append(bound)
bins.append(float('inf'))
frequency = []
for i in range(10):
    observed, expected = sum(res >= bins[i] and res < bins[i+1] for res in model_fit.resid), 1498*0.1
#     print('{:2d}'.format(observed), expected)
    frequency.append((observed, expected))
chi_square = sum([(x[0]-x[1])**2./x[1] for x in frequency])
chi_critical = chi2.ppf(0.95, 8) 
p_value = 1 - chi2.cdf(chi_square, 8)
print(chi_square, chi_critical, p_value)
#Chi Square Test ends