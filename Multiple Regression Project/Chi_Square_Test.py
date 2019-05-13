import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import chi2
import matplotlib.pyplot as plt
import statsmodels.formula.api as smapi
from __future__ import division

data = pd.read_csv('C:\\Users\\sshek\\Documents\\My Documents\\Fall 2018 Courses\\IOT Analytics\\Projects\\Multiple Regression\\sshekha4.csv', header=None, sep=",", names = ["X1", "X2", "X3", "X4", "X5", "Y"]);
print(data)
# For Simple Linear Regression
print("Performing for Simple Regression")
data.plot(kind="scatter", x="X1", y="Y")
simpleLinearMod = smapi.ols(formula = "Y ~ X1", data = data).fit()
simpleLinearMod.summary()
res_mean = 0; res_sd = simpleLinearMod.resid.std();
print("Residual Mean:", res_mean, "  and  Residual SD:", res_sd)
bins = [-float('inf')]
for i in range(9):
    p = (i+1)/10. ; z = norm.ppf(p) ; bound = res_mean + (z*res_sd);
    print(p, '{:6.4f}'.format(z), '{:6.4f}'.format(bound))
    bins.append(bound)
bins.append(float('inf'))
frequency = []
for i in range(10):
    observed, expected = sum(res >= bins[i] and res < bins[i+1] for res in simpleLinearMod.resid), 300*0.1
    print('{:2d}'.format(observed), expected)
    frequency.append((observed, expected))
chi_square = sum([(x[0]-x[1])**2./x[1] for x in frequency])
chi_critical = chi2.ppf(0.95, 7) # 7 are the degrees of freedom for bin size of 10 (1 for mean, 1 for SD, 1 for the last value)
p_value = 1 - chi2.cdf(chi_square, 7)
print(chi_square, chi_critical, p_value)

# For Multiple Linear Regression
print("Performing for Multiple Regression")
multipleLinearMod = smapi.ols(formula = "Y ~ X1+X2+X3+X4+X5", data = data).fit()
multipleLinearMod.summary()
res_mean = 0; res_sd = multipleLinearMod.resid.std();
print("Residual Mean:", res_mean, "  and  Residual SD:", res_sd)
bins = [-float('inf')]
for i in range(9):
    p = (i+1)/10. ; z = norm.ppf(p) ; bound = res_mean + (z*res_sd);
    print(p, '{:6.4f}'.format(z), '{:6.4f}'.format(bound))
    bins.append(bound)
bins.append(float('inf'))
frequency = []
for i in range(10):
    observed, expected = sum(res >= bins[i] and res < bins[i+1] for res in simpleLinearMod.resid), 300*0.1
    print('{:2d}'.format(observed), expected)
    frequency.append((observed, expected))
chi_square = sum([(x[0]-x[1])**2./x[1] for x in frequency])
chi_critical = chi2.ppf(0.95, 7) # 7 are the degrees of freedom for bin size of 10 (1 for mean, 1 for SD, 1 for the last value)
p_value = 1 - chi2.cdf(chi_square, 7)
print(chi_square, chi_critical, p_value)