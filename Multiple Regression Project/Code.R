library(tidyverse)
library(ggplot2)
library(corrplot)

# Set directory 
setwd("C:\\Users\\sshek\\Documents\\My Documents\\Fall 2018 Courses\\IOT Analytics\\Projects\\Multiple Regression")
options(max.print = 10000)
# Read data from CSV file
data <- read.table("sshekha4.csv", header=FALSE, sep=",")

#Task 1

# For V1
hist(data$V1)
mean(data$V1)
var(data$V1)
# For V2
hist(data$V2)
mean(data$V2)
var(data$V2)
#For V3
hist(data$V3)
mean(data$V3)
var(data$V3)
#For V4
hist(data$V4)
mean(data$V4)
var(data$V4)
#For V5
hist(data$V5)
mean(data$V5)
var(data$V5)

# Experimental Section Starts

#Removing outliers (using mahalanobis distance for anomaly detection)
#data.anomaly <- data
#data$MD <- mahalanobis(data[, c(1:5)], colMeans(data[, c(1:5)]), cov(data[, c(1:5)]))
#data.subset <- (data %>% filter(data$MD <= 10.0))
#data.noanomaly <- data.subset %>% select('V1', 'V2', 'V3', 'V4', 'V5', 'V6')

#Comparing the correlation matrix with and without anomalies
#cor(data.anomaly) # With anomalies
#cor(data.noanomaly) # Without anomalies

# Experimental Section Ends

# Checking for outliers (using boxplot for individual independent variables)
# first copying original Data to a new variable for later use and comparison
originalData <- data
# Removing Outliers
#For X1
data[! data$V1 %in% boxplot(data$V1)$out, ]
plot(data$V1, data$V6)
#For X2
data[! data$V2 %in% boxplot(data$V2)$out, ]
plot(data$V2, data$V6)
data <- data[! data$V2 %in% boxplot(data$V2)$out, ]
#For X3
data[! data$V3 %in% boxplot(data$V3)$out, ]
plot(data$V3, data$V6)
data <- data[! data$V3 == max(data$V3), ]
#For X4
data[! data$V4 %in% boxplot(data$V4)$out, ]
plot(data$V4, data$V6)
data <- data[! data$V4 %in% boxplot(data$V4)$out, ]
#For X5
data[! data$V5 %in% boxplot(data$V5)$out, ]
plot(data$V5, data$V6)
data <- data[! data$V5 == min(data$V5), ]
nrow(data)
#Correlation Matrix
cor(originalData)
cor(data)

# Task 2
simpleLinearMod <- lm(data$V6 ~ data$V1, data=data)
plot(data$V6 ~ data$V1)
abline(simpleLinearMod)
# Below is the smothing curve that checks for a better fit.
lines(lowess(data$V6 ~ data$V1))
summary(simpleLinearMod)
#qqplot
qqplot(residuals(simpleLinearMod), rnorm(nrow(data), mean = 0, sd = 141.1))
#histogram
simpleLinearMod$residuals
hist(simpleLinearMod$residuals)
plot(simpleLinearMod$fitted.values, simpleLinearMod$residuals)
#Going for higher order polynomials to check the fit
higherOrderMod <- lm(data$V6 ~ data$V1 + I(data$V1^2), data=data)
summary(higherOrderMod)
#P-value for the linear term > 0.05 suggests that it is not significant. Hence, removing it
higherOrder2Mod <- lm(data$V6 ~ I(data$V1^2), data=data)
plot(simpleLinearMod$fitted.values, higherOrder2Mod$residuals)
summary(higherOrder2Mod)
summary(simpleLinearMod)

#Task 3 - Multivariable Linear Regression
multipleMod <- lm(data$V6 ~ data$V1 + data$V2 + data$V3 + data$V4 + data$V5, data=data)
summary(multipleMod)
plot(multipleMod)

#qqplot
qqplot(residuals(multipleMod), rnorm(nrow(data), mean = 0, sd = 38.06))
#histogram
residuals(multipleMod)
hist(residuals(multipleMod))
plot(multipleMod$fitted.values, multipleMod$residuals)