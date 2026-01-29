# **Introduction: General and Generalized Linear Regression Models**

**Before starting, this solution can be run in Colab without changing the ___"Change runtime type"___**
**We can run the solution just in python.**

```R
# activate R magic
%load_ext rpy2.ipython
%reload_ext rpy2.ipython
```

Keep **'%%R'** for every code cell ONLY WHEN USING R CODE.
```R
%%R
library(readr)
library(ggplot2)
```
_If we want to run multiple varaible from different data frames with the same mask each time the data frame so the computer doesn't confused it by either using the **deattch()** function or use the **$** symbol for example **variable$dataFrame**._ 

Create another code cell and this will allow us to call the libraries in python.
```Python
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stats
from IPython.display import display, Math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
## Example 1
A small-scale clinical trial is conducted to compare the efficacy of two drugs (A and B) in the reduction of excess body weight. Drug (A or B), age, and gender were recorded at the baseline. The percent excess body weight loss (EWL) was recorded 3 months into the study. The data are available on 32 subjects:

**READ CSV file Colab in R**
```R
# Read .cvs file into Colab in R code from GitHub
%%R
weight.lost.data <- read_csv('https://raw.githubusercontent.com/coronasanchez/LACC-MATH-239/main/HW1/Exercise1.2Data.csv')
head(weight.lost.data)
summary(weight.lost.data)
attach(weight.lost.data)
```
**Plot the histogram for the scores (y) and conduct the normality tests**
```R
# Plot the historgram for the scores(y) and conduct the normal tests
%%R
hist( weight.lost.data$EWL,
      main = 'Weight Lost',
      col = c('blue', 'cyan'),
      ylim = c(0,10),
      xlim = c(0,35),
      ylab = 'COUNTS',
      xlab = 'EWL' )
```
<p align="center">
	<img width="350" height="auto" alt="image" src="https://github.com/user-attachments/assets/4579f7d8-feed-45f5-ba73-813c0a662df4" />
</p>
```R
%%R
EWL.hist <- ggplot(weight.lost.data, aes(x = EWL)) +
  geom_histogram(aes(y = after_stat(density)), bins = 12, fill ="gray",color = "black") +
    stat_function(fun = dnorm, args = list(mean = mean(weight.lost.data$EWL), sd = sd(weight.lost.data$EWL)), color = "red", linewidth = 1.2) +
      labs(title = "Weight lost with Normal Distribution Line with fitted line", x = "EWL", y = "Relative Frequency") +
        theme_bw()

print(EWL.hist)

# Shapiro-Wilk test
shapiro.test(weight.lost.data$EWL)
```
_Cell output:_
``` 
	Shapiro-Wilk normality test

data:  weight.lost.data$EWL
W = 0.97424, p-value = 0.6234

```
<img width="480" height="480" alt="image" src="https://github.com/user-attachments/assets/fcec5982-b409-4f63-ad58-e4687a857d16" />

**Fit a general linear model.** 
```R
%%R
# specifying reference levels
# Convert categorical variables to factors
weight.lost.data$drug <-as.factor(weight.lost.data$drug)
weight.lost.data$gender <- as.factor(weight.lost.data$gender)

# Specify reference levels for the categorical variables
drug.rel <- relevel(weight.lost.data$drug, ref = "A")
gender.rel <- relevel(weight.lost.data$gender, ref = "M")

# Fit the generalized linear model
fitted.model <- glm(EWL ~ drug.rel + age + gender.rel,
                    data = weight.lost.data,
                    family = gaussian(link = identity))

# Print the summary of the model
summary(fitted.model)
```
_Cell output:_
```
Call:
glm(formula = EWL ~ drug.rel + age + gender.rel, family = gaussian(link = identity), 
    data = weight.lost.data)

Coefficients:
            Estimate Std. Error t value Pr(>|t|)  
(Intercept)   9.2146     5.6981   1.617   0.1171  
drug.relB     4.8103     1.9988   2.407   0.0229 *
age           0.1102     0.1140   0.967   0.3420  
gender.relF   2.7235     1.9952   1.365   0.1831  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 31.44133)

    Null deviance: 1144.12  on 31  degrees of freedom
Residual deviance:  880.36  on 28  degrees of freedom
AIC: 206.88

Number of Fisher Scoring iterations: 2
```
Reduce Model: <br>
$\widehat{\mathbb{E}}(EWL)=9.2146+4.8103\cdot drug B$

**Model fits the data well?**
```R
%%R
# PART 4: Model fits the data well?
logLik(fitted.model)
%%R
# checking model fit
#outputting estimated sigma
cat("sigma: ", sigma(fitted.model), "\n")

null.model <-
  glm(EWL ~ 1, data = weight.lost.data, family = gaussian(link = identity))
cat("devidnace =", deviance<- -2*(logLik(null.model)-logLik(fitted.model)), "\n")

# get the total number of coefficients
num_coeff_EWL_df <- length(coef(fitted.model))
print(num_coeff_EWL_df)

cat("p-value =", p.value <- pchisq(deviance, df= num_coeff_EWL_df - 1, lower.tail=FALSE), "\n") # df num of columns
```
Let us predict using the fitted model to predict if the percent decrease in excess body weight for a 35-year-old male who is taking drug A?

**${EWL}^0$ = $9.2146+0.1102\cdot (35)=13.0716 $**
```R
%%R
# using fitted model for prediction
ewl_prediction <-predict(fitted.model, data.frame(drug.rel="A", age=35, gender.rel="M"))
cat("Predicted EWl:", ewl_prediction)
```
_Cell output:_
```
Predicted EWl: 13.07178
```

## Example 2
A person is thinking of buying a new car. He conducts an online search and collects information on makes and models that he likes. He conjectures that the following car characteristics may potentially influence its price: body style (coupe, hatchback, or sedan), country of manufacture (USA, Germany, or Japan), highway mileage (in mpg), number of doors (2 or 4), and whether the interior is leather or not. The data for these variables and the price (in U.S. dollars) are given below for 27 cars.

**READ CSV file Colab in R**
```R
# Read .cvs file into Colab in R code from GitHub
%%R
car.decision.data <- read_csv('https://raw.githubusercontent.com/coronasanchez/LACC-MATH-239/main/HW1/Exercise1.3Data.csv')
head(car.decision.data)
summary(car.decision.data)
attach(car.decision.data)
```

**Plot the histogram for the scores (y) and conduct the normality tests**
```R
%%R
hist(price,
      col = c('blue', 'cyan'),
      xlab = 'Price 1k',
      ylim = c(0, 10),
      ylab = 'COUNTS',
      main = 'Total Number of Car Price')
```
<p align = "center">
	<img width="350" height="auto" alt="image" src="https://github.com/user-attachments/assets/3682de28-7e95-4b44-8564-dc3afd010563" />
</p>

```R
%%R
# Reduce the car price by the factor of 1000.
library(ggplot2)

price1k_hist <- ggplot(car.decision.data, aes(x = price/1000)) +
  geom_histogram(aes(y = after_stat(density)), bins = 10, fill ="gray",color = "black") +
    stat_function(fun = dnorm, args = list(mean = mean(car.decision.data$price/1000), sd = sd(car.decision.data$price/1000)), color = "red", linewidth = 1) +
      labs(title = "Car Price with Normal Distribution Line with fitted line", x = "Price", y = "Relative Frequency") +
        theme_bw()

print(price1k_hist)

shapiro.test(car.decision.data$price/1000)
```
_Cell output:_
```
	Shapiro-Wilk normality test

data:  car.decision.data$price/1000
W = 0.95482, p-value = 0.28
```
<p align = "center">
	<img width="350" height="auto" alt="image" src="https://github.com/user-attachments/assets/8352e247-3e7e-44f3-9cb1-832f44ffed7f" />
</p>

**Fit a general linear model**
```R
%%R
# specifying reference levels
# Convert categorical variables to factors
weight.lost.data$drug <-as.factor(weight.lost.data$drug)
weight.lost.data$gender <- as.factor(weight.lost.data$gender)

# Specify reference levels for the categorical variables
drug.rel <- relevel(weight.lost.data$drug, ref = "A")
gender.rel <- relevel(weight.lost.data$gender, ref = "M")

# Fit the generalized linear model
fitted.model <- glm(EWL ~ drug.rel + age + gender.rel,
                    data = weight.lost.data,
                    family = gaussian(link = identity))

# Print the summary of the model
summary(fitted.model)
```
_Cell output:_
```
Call:
glm(formula = EWL ~ drug.rel + age + gender.rel, family = gaussian(link = identity), 
    data = weight.lost.data)

Coefficients:
            Estimate Std. Error t value Pr(>|t|)  
(Intercept)   9.2146     5.6981   1.617   0.1171  
drug.relB     4.8103     1.9988   2.407   0.0229 *
age           0.1102     0.1140   0.967   0.3420  
gender.relF   2.7235     1.9952   1.365   0.1831  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 31.44133)

    Null deviance: 1144.12  on 31  degrees of freedom
Residual deviance:  880.36  on 28  degrees of freedom
AIC: 206.88

Number of Fisher Scoring iterations: 2
```
Reduce Model: <br> $\widehat{\mathbb{E}} (price/1000) = 5.1353 + 6.4107 ⋅ (sedan) + 12.1757 ⋅ (leather) $

**Model Fits the data well?**
```R
%%R
logLik(fitted.model)

%%R
# checking model fit
#outputting estimated sigma
cat("sigma: ", sigma(fitted.model), "\n")

null.model <-
  glm(EWL ~ 1, data = weight.lost.data, family = gaussian(link = identity))
cat("devidnace =", deviance<- -2*(logLik(null.model)-logLik(fitted.model)), "\n")

# get the total number of coefficients
num_coeff_EWL_df <- length(coef(fitted.model))
print(num_coeff_EWL_df)

cat("p-value =", p.value <- pchisq(deviance, df= num_coeff_EWL_df - 1, lower.tail=FALSE), "\n") # df num of columns
```
**Fitted model:** $\widehat{\mathbb{E}} (price/1000) = 5.1353 + 6.4107 ⋅ (sedan) + 12.1757 ⋅ (leather) $

_Cell output:_
```
'log Lik.' -98.4395 (df=5)

sigma:  5.607257 
devidnace = 8.386158 
[1] 4
p-value = 0.03867005 
```
According to the model, what is the predicted percent decrease in excess body weight for a 35-year-old male who is taking drug A?

**${EWL}^0$ = $9.2146+0.1102\cdot (35)=13.0716 $**
```R
%%R
# using fitted model for prediction
ewl_prediction <-predict(fitted.model, data.frame(drug.rel="A", age=35, gender.rel="M"))
cat("Predicted EWl:", ewl_prediction)
```
__Cell output:__
```
Predicted Car Price: $ 37071.14
```

## Example 3
Fifty people were surveyed randomly regarding the number of hours of quality sleep they normally get per night. Additional measurements on surveyed participants were age (in years), gender(M/F), number of minutes per day spent having personal quiet time, number of children under 5 years of age in the household, daily stress level (on a scale of 1 to 10), current job status(full/part/unemployed/student), number of physical activities per week, and number of months since last vacation or a weekend get-away.

**Read CSV file into Colab**
```R
%%R
sleep.quality.data <- read_csv('https://raw.githubusercontent.com/coronasanchez/LACC-MATH-239/main/HW1/Exercise1.4Data.csv')
head(sleep.quality.data)

summary(sleep.quality.data)

attach(sleep.quality.data)
```

**Plot the histogram for the scores (y) and conduct the normality tests**
```R
%%R
hist(Sleephours,
      col = c('blue', 'cyan'),
      xlab = 'Sleep Hours',
      ylab = 'COUNTS',
      ylim = c(0, 12),
      main = 'Total Hour of Sleep')
```
<p align = "center">
	<img width="350" height="auto" alt="image" src="https://github.com/user-attachments/assets/f3714023-668b-43b9-9f05-b1c103dd5e01" />
</p>

```R
%%R

sleep_hist <- ggplot(sleep.quality.data, aes(x = Sleephours)) +
  geom_histogram(aes(y = after_stat(density)), bins = 15, fill ="gray",color = "black") +
    stat_function(fun = dnorm, args = list(mean = mean(sleep.quality.data$Sleephours), sd = sd(sleep.quality.data$Sleephours)), color = "red", linewidth = 1) +
      labs(title = "Sleep Hours with Normal Distribution Line with fitted line", x = "Sleep Hours", y = "Relative Frequency") +
        theme_bw()

print(sleep_hist)

shapiro.test(sleep.quality.data$Sleephours)
```
_Cell output:_
```
 
	Shapiro-Wilk normality test

data:  sleep.quality.data$Sleephours
W = 0.98284, p-value = 0.6762
```
<p align = "center">
	<img width="350" height="auto" alt="image" src="https://github.com/user-attachments/assets/bc128cef-cfeb-4b56-bd1a-62eff466b81f" />
</p>

**Fit a general linear model**
```R
# specifying reference levels
# Convert categorical variables to factors
%%R
sleep.quality.data$Gender <- as.factor(sleep.quality.data$Gender)
sleep.quality.data$JobStatus <- as.factor(sleep.quality.data$JobStatus)

# Specify reference levels for the categorical variables
sleep.gender.refl <- relevel(sleep.quality.data$Gender, ref = 'F')
jobStatus.refl <- relevel(sleep.quality.data$JobStatus, ref = 'full')

# Fit the generalized linear model
sleep.fitted.model <- glm(Sleephours ~ Age + sleep.gender.refl + QuietTime + NChildren + StressLevel + jobStatus.refl + NActivities + PastVac, data = sleep.quality.data, family = gaussian(link = identity))

# Print the summary of the model
summary(sleep.fitted.model)
```
_Cell output:_
```

Call:
glm(formula = Sleephours ~ Age + sleep.gender.refl + QuietTime + 
    NChildren + StressLevel + jobStatus.refl + NActivities + 
    PastVac, family = gaussian(link = identity), data = sleep.quality.data)

Coefficients:
                       Estimate Std. Error t value Pr(>|t|)    
(Intercept)            6.826002   0.798388   8.550 1.78e-10 ***
Age                   -0.003656   0.010494  -0.348  0.72943    
sleep.gender.reflM     0.356815   0.241401   1.478  0.14741    
QuietTime              0.007421   0.003238   2.292  0.02738 *  
NChildren              0.120419   0.123020   0.979  0.33368    
StressLevel           -0.139828   0.060734  -2.302  0.02674 *  
jobStatus.reflpart     1.048386   0.360976   2.904  0.00603 ** 
jobStatus.reflstudent  0.628623   0.493437   1.274  0.21021    
jobStatus.reflunempl   0.381840   0.323501   1.180  0.24501    
NActivities            0.020373   0.039031   0.522  0.60465    
PastVac                0.005046   0.019222   0.263  0.79430    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 0.6672346)

    Null deviance: 54.322  on 49  degrees of freedom
Residual deviance: 26.022  on 39  degrees of freedom
AIC: 133.24

Number of Fisher Scoring iterations: 2
```
Reduce Model: <br> $\widehat{\mathbb{E}}(Sleep Quality) = 6.826002 + 0.007421 ⋅ (Quiet Time) - 0.139828 · (Stress Level) + 1.048386 ⋅ (PartTime)$

**Model Fits the data well?**
```R
%%R
# PART 4: Model fits the data well?
print(logLik(sleep.fitted.model))

# out estimated sigma
cat("sigma:", sigma(sleep.fitted.model), "\n")

# checking model fit
sleep.null.model <- glm(Sleephours ~ 1, data = sleep.quality.data, family = gaussian(link = identity))

sleep.deviance <- -2*(logLik(sleep.null.model)-logLik(sleep.fitted.model))
cat("deviance:", sleep.deviance , "\n")

num_coeff_sq_df <- length(coef(sleep.fitted.model))
print(num_coeff_sq_df)

sleep.p.value <- pchisq(sleep.deviance, df = num_coeff_sq_df - 1, lower.tail=FALSE)
cat("p-value:",sleep.p.value) # df num of columns
```

Find the predicted number of hours of night's sleep that a 30-year-old full-time mom of three children under the age of five has, if she gets 10 minutes a day for herself, walks to the park with her kids every day of the week, estimates her stress level as 7, and who hasn't gotten any vacation for one year.

$Sleephours^0 = 6.826002 - 0.003656 ⋅ (30) + 0.007421 ⋅ (10) + 0.120419 · (3) - 0.139828 · (7) + 0.381840 ⋅ (1) + 0.020373 ⋅ (7)$ <br>
$Sleephours^0 = 6.696848$

```R
%%R
print(predict(sleep.fitted.model, data.frame(Age=30, sleep.gender.refl='F', QuietTime=10, NChildren=3, StressLevel=7, jobStatus.refl='unempl', NActivities=7, PastVac=0)))
```
_Cell output:_
```
       1 
6.697449 
```

## Example 4
A cardiologist conducts a study to find out what factors are good predictors of elevated heart rate (HR) in her patients. She measures heart rate at rest in 30 patients on their next visit and obtains from the medical charts additional information on their age, gender, ethnicity(Black, Hispanic, or White), body mass index (BMI), and the number of currently taken heart medications. She also obtains the air quality index (AQI) for the area of residence of her patients (unhealthy/moderate/good).

**Read CSV file into Colab**
```R
%%R
# Read .cvs file into Colab in R code from GitHub
cardiologist.HR.data <- read_csv('https://raw.githubusercontent.com/coronasanchez/LACC-MATH-239/main/HW1/Exercise1.6Data.csv', show_col_types = FALSE)

head(cardiologist.HR.data)

summary(cardiologist.HR.data)

attach(cardiologist.HR.data)
```

**Plot the histogram for the scores (y) and conduct the normality tests**
```R
%%R
hist(cardiologist.HR.data$HR ,
      col = c('blue', 'cyan'),
      xlab = 'HR',
      xlim = c(70, 120),
      ylab = 'COUNTS',
      ylim = c(0, 6),
      main = 'Total Count of HR')
```
<p align = "center">
	<img width="350" height="auto" alt="image" src="https://github.com/user-attachments/assets/c6e6eff6-7c0f-43bc-9199-4df2148fe24a" />
</p>

```R
%%R
library(ggplot2)

HR.hist <- ggplot(data = cardiologist.HR.data, aes(x = HR)) +
  geom_histogram(aes(y = after_stat(density)), bins = 8, fill ="gray",color = "black") +
    stat_function(fun = dnorm, args = list(mean = mean(HR), sd = sd(HR)), color = "red", linewidth = 1) +
      labs(title = "HR with Normal Distribution Line with fitted line", x = "Sleep Hours", y = "Relative Frequency") +
        theme_bw()

print(HR.hist)
shapiro.test(HR)
```
_Cell output:_
```
 
	Shapiro-Wilk normality test

data:  HR
W = 0.93047, p-value = 0.05054
```
<p align = "center">
	<img width="480" height="480" alt="image" src="https://github.com/user-attachments/assets/049a4b51-5caf-4227-8fda-3b635feb31c5" />
</p>

**Fit a general linear model**
```R
%%R
# specifying reference levels
# Convert categorical variables to factors
cardiologist.HR.data$gender <- as.factor(cardiologist.HR.data$gender)
cardiologist.HR.data$ethnicity <- as.factor(cardiologist.HR.data$ethnicity)
cardiologist.HR.data$AQI <- as.factor(cardiologist.HR.data$AQI)

# Specify reference levels for the categorical variables
gender.HR.refl <- relevel(cardiologist.HR.data$gender, ref = 'M')
ethnicity.refl <- relevel(cardiologist.HR.data$ethnicity, ref = 'Hispanic')
AQI.refl <- relevel(cardiologist.HR.data$AQI, ref = 'good')

# Fit the generalized linear model
HR.fitted.model <-
  glm(data=cardiologist.HR.data,
      HR ~ age + gender.HR.refl + ethnicity.refl + BMI + nmeds + AQI.refl,
      family=gaussian(link=identity))

# Print the summary of the model
summary(HR.fitted.model)
```
_Cell output:_
```

Call:
glm(formula = HR ~ age + gender.HR.refl + ethnicity.refl + BMI + 
    nmeds + AQI.refl, family = gaussian(link = identity), data = cardiologist.HR.data)

Coefficients:
                    Estimate Std. Error t value Pr(>|t|)   
(Intercept)         38.01638   12.24005   3.106  0.00535 **
age                  0.65033    0.17599   3.695  0.00134 **
gender.HR.reflF      7.10311    2.82173   2.517  0.02002 * 
ethnicity.reflBlack  7.53509    3.46094   2.177  0.04102 * 
ethnicity.reflWhite  2.26328    3.33411   0.679  0.50466   
BMI                  0.04306    0.38543   0.112  0.91210   
nmeds                0.43836    1.42454   0.308  0.76133   
AQI.reflmoderate    10.85963    3.22023   3.372  0.00288 **
AQI.reflunhealthy   14.16737    3.81333   3.715  0.00128 **
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 51.28116)

    Null deviance: 4537.2  on 29  degrees of freedom
Residual deviance: 1076.9  on 21  degrees of freedom
AIC: 212.56

Number of Fisher Scoring iterations: 2
```
Reduce Model: <br> $\widehat{\mathbb{E}}$ $(HR) = 38.01638 + 0.65033 ⋅ (age) + 7.10311 ⋅ (female) + 7.53509 ⋅ (Black) + 10.85963 ⋅ (AQI Moderate) + 14.16736 ⋅ (AQI Healthy)$

**Model Fits the data well ?**
```R
%%R
print(logLik(HR.fitted.model))

# out estimated sigma
cat("sigma", sigma(HR.fitted.model), "\n")
# checking model fit
HR.null.model <- glm(cardiologist.HR.data$HR ~ 1, data = cardiologist.HR.data, family = gaussian(link = identity))

HR.deviance<- -2*(logLik(HR.null.model)-logLik(HR.fitted.model))
cat("deviance:", HR.deviance, "\n")

num_coeff_HR_df <- length(coef(HR.fitted.model))
print(num_coeff_HR_df)

HR.p.value <- pchisq(HR.deviance, df = num_coeff_HR_df - 1, lower.tail=FALSE)
cat("p-value:", HR.p.value, "\n")
```
Compute the predicted heart rate of a 50-year-old Hispanic male who has a BMI of 20, is not taking any heart medications, and resides in an area with moderate air quality.

$HR^0 = 38.01638 + 0.65033 ⋅ (50) + 7.10311 ⋅ (0) + 7.53509 ⋅ (0) + 2.26328 · (0) + 0.04306 · (20) + 0.43836 ⋅ (0) + 10.85963 ⋅ (1) + 14.16736 ⋅ (0)$ <br>
$HR^0 = 82.25371$

```R
HR_predict_data = pd.DataFrame({'age': [50], 'gender': ['M'], 'ethnicity':['Hispanic'], 'BMI':[20], 'nmeds':[0], 'AQI':['moderate']})
# predict_data

HR_prediction = HR_model.predict(HR_predict_data)
print(f"Predicted Heart Rate: {HR_prediction[0]:.2f}")
```
_Cell output:_
```
Predicted Heart Rate: 82.25
```
