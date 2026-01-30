# Regression Models for Response for Binary Response
* Binary Logistic Regression
* Probit Model
* Complementary Log-Log Model (CLL)

Before starting, run the code below to import all necessary functions and libraries.
```R
library(readr)
library(ggplot2)
```

## Exercise 1
Dermatologists in a hospital study patients with acute psoriasis, a skin disease. They would like to know whether medication A is more effective in relieving the symptoms of psoriasis than medication B. The data are retrospectively collected on 30 patients. The variables are gender (M/F), age (in years), medication(A/B), and response (1=relief, 0=no relief). 
**Read CSV file into Colab**
```R
SkinDisease.data <- read_csv('https://raw.githubusercontent.com/coronasanchez/LACC-MATH-239/main/HW3/Exercise3.2Data.csv')

attach(SkinDisease.data)
head(SkinDisease.data)
```

**Fit a general linear model**
```R
# Rescaling Variables and specifying reference categories
SkinDisease.gender.relf <- relevel(factor(SkinDisease.data$gender, order = FALSE), ref = 'F')
SkinDisease.medication.relf <- relevel(factor(SkinDisease.data$medication, order = FALSE), ref = 'B')
```

* Binary Logistic Regression Model
```R
# Binary Logistic Regression Model
# Fit the generalized linear model to transformed response
relief_bi.fitted.model <- glm(relief ~ SkinDisease.gender.relf + age + SkinDisease.medication.relf, data = SkinDisease.data, family = binomial(link = logit))

# Print the summary of the model
summary(relief_bi.fitted.model)

# outputting estimated sigma
# cat("\nsigma:\t", sigma(bmi.fitted.model))
```
_Cell Output:_
```

Call:
glm(formula = relief ~ SkinDisease.gender.relf + age + SkinDisease.medication.relf, 
    family = binomial(link = logit), data = SkinDisease.data)

Coefficients:
                             Estimate Std. Error z value Pr(>|z|)  
(Intercept)                  -6.79921    3.11430  -2.183   0.0290 *
SkinDisease.gender.relfM      3.17132    1.47097   2.156   0.0311 *
age                           0.17131    0.08691   1.971   0.0487 *
SkinDisease.medication.relfA  3.81641    1.54617   2.468   0.0136 *
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 36.652  on 29  degrees of freedom
Residual deviance: 21.008  on 26  degrees of freedom
AIC: 29.008

Number of Fisher Scoring iterations: 6
```
**Reduced Binary Fitted Model:** $\frac{\widehat{\pi} (relief)}{1-\widehat{\pi}(relief)}$ $=e^{3.17132 ⋅ (male) + 0.17131 ⋅ (age) + 3.81641 ⋅ (MedicationA) - 6.79921}$  

Extracting AIC, AICC, and BIC.
```R
# Binary Logistic Regression Model
relief_bi.numcoeff <- length(coef(relief_bi.fitted.model))
cat("p = ", relief_bi.numcoeff)
cat('\nn = ', SkinDisease.n)

# extracting AICC and BIC for fitted model
cat("\n\n\tBinary Logistic\n")
cat("AIC  = \t", AIC(relief_bi.fitted.model))
# -2logLik(fitted.model)+2*p*n / (n- p -1)
SkinDisease_bi.AICC <- -2 * logLik(relief_bi.fitted.model) + 2 * relief_bi.numcoeff * SkinDisease.n / (SkinDisease.n - relief_bi.numcoeff - 1)
cat('\nAICC = \t', SkinDisease_bi.AICC)

SkinDisease_bi.BIC <- BIC(relief_bi.fitted.model)
cat("\nBIC  = \t", SkinDisease_bi.BIC)
```
_Cell Output:_
```
p =  4
n =  30

	Binary Logistic
AIC  = 	 29.00842
AICC = 	 30.60842
BIC  = 	 34.6132
```

* Probit Model
```R
# Probit Model
# Fit the generalized linear model to transformed response
relief_probit.fitted.model <- glm(
  relief ~ SkinDisease.gender.relf + age + SkinDisease.medication.relf,
  data = SkinDisease.data,
  family = binomial(link = probit)
  )

# Print the summary of the model
summary(relief_probit.fitted.model)

# outputting estimated sigma
# cat("\nsigma:\t", sigma(bmi.fitted.model))
```

_Cell Output:_
```
 
Call:
glm(formula = relief ~ SkinDisease.gender.relf + age + SkinDisease.medication.relf, 
    family = binomial(link = probit), data = SkinDisease.data)

Coefficients:
                             Estimate Std. Error z value Pr(>|z|)   
(Intercept)                  -4.07825    1.76524  -2.310  0.02087 * 
SkinDisease.gender.relfM      1.92301    0.80597   2.386  0.01703 * 
age                           0.10260    0.04969   2.065  0.03892 * 
SkinDisease.medication.relfA  2.23351    0.85603   2.609  0.00908 **
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 36.652  on 29  degrees of freedom
Residual deviance: 20.755  on 26  degrees of freedom
AIC: 28.755

Number of Fisher Scoring iterations: 8
```
** Reduced Probit Fitted Model:** $\phi^{-1} ⋅ \widehat{\pi}(relief)$ $=1.92301 ⋅ (male) + 0.10260 ⋅ (age) + 2.23351 ⋅ (MedicationA) - 4.07825$

Extracting AIC, AICC, and BIC. 
```R
# Probit Model
relief_probit.numcoeff <- length(coef(relief_probit.fitted.model))
cat("p = ", relief_bi.numcoeff)
cat('\nn = ', SkinDisease.n)

cat('\n\n\tProbit Model\n')
# extracting AICC and BIC for fitted model
cat('AIC  =\t', AIC(relief_probit.fitted.model))
relief_probit.AICC <- -2 * logLik(relief_probit.fitted.model) + 2 * relief_probit.numcoeff * SkinDisease.n / ( SkinDisease.n - relief_probit.numcoeff - 1)
cat('\nAICC =\t', relief_probit.AICC)
cat('\nBIC  =\t', BIC(relief_probit.fitted.model))
```
_Cell Output:_
```
p =  4
n =  30

	Probit Model
AIC  =	 28.75491
AICC =	 30.35491
BIC  =	 34.3597
```

* Complementary Log-Log Model
```R
# Complementary Log-Log Model
# Fitting complementary log-log model
relief_cloglog.fitted.model <- glm(
  relief ~ SkinDisease.gender.relf + age + SkinDisease.medication.relf,
  data = SkinDisease.data,
  family = binomial(link = cloglog)
)
# Print the summary of the model
summary(relief_cloglog.fitted.model)
```
_Cell Output:_
```
 Warning message:
“glm.fit: fitted probabilities numerically 0 or 1 occurred”

Call:
glm(formula = relief ~ SkinDisease.gender.relf + age + SkinDisease.medication.relf, 
    family = binomial(link = cloglog), data = SkinDisease.data)

Coefficients:
                             Estimate Std. Error z value Pr(>|z|)  
(Intercept)                  -4.73174    2.12142  -2.230   0.0257 *
SkinDisease.gender.relfM      2.15577    0.87247   2.471   0.0135 *
age                           0.10686    0.05708   1.872   0.0612 .
SkinDisease.medication.relfA  2.23606    0.96039   2.328   0.0199 *
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 36.652  on 29  degrees of freedom
Residual deviance: 20.612  on 26  degrees of freedom
AIC: 28.612

Number of Fisher Scoring iterations: 10
```
**Reduced Complementary Log-Log Fitted Model:** $1 - \widehat{\pi}(relief) = e^{-e^{1.92301 ⋅ (male) + 0.10260 ⋅ (age) + 2.23351 ⋅ (MedicationA) - 4.07825}}$

Extracting AIC, AICC, and BIC. 
```R
# Complementary Log-Log Model
relief_cll.numcoeff <- length(coef(relief_cloglog.fitted.model))
cat("p = ", relief_cll.numcoeff)
cat('\nn = ', SkinDisease.n)

cat('\n\tCLog Log\n')
# extracting AICC and BIC for fitted model
cat('AIC  =\t', AIC(relief_cloglog.fitted.model))
relief_cloglog.AICC <- -2 * logLik(relief_cloglog.fitted.model) + 2 * relief_cll.numcoeff * SkinDisease.n / ( SkinDisease.n - relief_cll.numcoeff - 1)
cat('\nAICC =\t', relief_cloglog.AICC)
cat('\nBIC  =\t', BIC(relief_cloglog.fitted.model))
```
_Cell Output:_
```
p =  4
n =  30
	CLog Log
AIC  =	 28.61248
AICC =	 30.21248
BIC  =	 34.21727
```

```R
# extracting AICC and BIC for fitted model
cat("Binary Logistic\t\tProbit Model \t\tCLog Log")
cat("\nAIC  = \t", AIC(relief_bi.fitted.model), '\tAIC  =\t', AIC(relief_probit.fitted.model), '\tAIC  =\t', AIC(relief_cloglog.fitted.model))
cat('\nAICC = \t', SkinDisease_bi.AICC, '\tAICC =\t', relief_probit.AICC, '\tAICC =\t', relief_cloglog.AICC)
cat("\nBIC  = \t", SkinDisease_bi.BIC, '\tBIC  =\t', BIC(relief_probit.fitted.model), '\tBIC  =\t', BIC(relief_cloglog.fitted.model))
```
_Cell Output:_
_Select the best Regression Model base on the smallest AIC, AICC, and BIC._ 
```
Binary Logistic		Probit Model 		CLog Log
AIC  = 	 29.00842 	AIC  =	 28.75491 	AIC  =	 28.61248
AICC = 	 30.60842 	AICC =	 30.35491 	AICC =	 30.21248
BIC  = 	 34.6132 	BIC  =	 34.3597 	BIC  =	 34.21727
```

**Model Fits the data well?**
* Binary Logistic Regression Model
```R
# Binary Logistic Regression Model
# Checking model fit
cat("\tBinary Logistic\n")
relief_bi.null.model <- glm(relief ~ 1, data = SkinDisease.data, family = binomial(link=logit))

relief_bi.deviance <- -2*(logLik(relief_bi.null.model)-logLik(relief_bi.fitted.model))

cat("deviance =\t", relief_bi.deviance)

relief_bi.pvalue <- pchisq(relief_bi.deviance, df = relief_bi.numcoeff - 1, lower.tail = FALSE)
cat("\np-value =\t", relief_bi.pvalue)
```
_Cell Output:_
```
	Binary Logistic
deviance =	 15.64344
p-value =	 0.001341726
```
* Probit Model
```R
# Probit Model
# checking model fit
cat('\tProbit Model\n')
relief_probit.null.model <- glm(
  relief ~ 1,
  data = SkinDisease.data,
  family = binomial(link = probit)
)

relief_probit.deviance <- -2*(logLik(relief_probit.null.model)-logLik(relief_probit.fitted.model))
cat("deviance =", relief_probit.deviance)

relief_pvalue <- pchisq(relief_probit.deviance, df = relief_probit.numcoeff - 1, lower.tail = FALSE)
cat("\np-value =", relief_pvalue)
```
_Cell Output:_
```
	Probit Model
deviance = 15.89695
p-value = 0.001190503
```
* Complementary Log-Log Model
```R
# CLogLog Model
#checking model fit
cat('\tClog Log Model\n')
relief_cloglog.null.model <- glm(relief ~ 1,data = SkinDisease.data, family = binomial(link = cloglog))
relief_cloglog.deviance <- -2 * (logLik(relief_cloglog.null.model) - logLik(relief_cloglog.fitted.model))
cat("deviance = ", relief_cloglog.deviance)

relief_cloglog.pvalue <- pchisq(relief_cloglog.deviance, df = relief_cll.numcoeff - 1, lower.tail = FALSE)
cat("\np-value = ", relief_cloglog.pvalue)
```
_Cell Output:_
```
	Clog Log Model
deviance =  16.03937
p-value =  0.0011131
```
**Using the fitted model to predict**
Find the predicted probability of relief from psoriasis for a 50-year-old woman who is administered the medication A treatment.

* Binary Logistic Regression Model
```R
# Binary Logistic Regression Model
# Using fitted moel for predition
cat('\tBinary Model: Prediction\n')
relief_bi.predict <- predict(relief_bi.fitted.model, data.frame(
  SkinDisease.gender.relf = 'F', age = 50 , SkinDisease.medication.relf = 'A'
), type = 'response')
cat(relief_bi.predict)
```
$P(relief)^0 = \frac{e^{3.17132 ⋅ (male) + 0.17131 ⋅ (age) + 3.81641 ⋅ (MedicationA) - 6.79921}}{1+e^{3.17132 ⋅ (male) + 0.17131 ⋅ (age) + 3.81641 ⋅ (MedicationA) - 6.79921}}$
$P(relief)^0 = \frac{e^{3.17132 ⋅ (0) + 0.17131 ⋅ (50) + 3.81641 ⋅ (1) - 6.79921}}{1+e^{3.17132 ⋅ (0) + 0.17131 ⋅ (50) + 3.81641 ⋅ (1) - 6.79921}}$
$P(relief)^0 = 0.99625$

_Cell Output:_
```
	Binary Model: Prediction
0.9962508
```

* Probit Model
```R
# Probit Model
# checking model fit
cat('\tProbit Model: Prediction\n')
relief_probit.pred <- predict(
  relief_probit.fitted.model,
  data.frame(SkinDisease.gender.relf = 'F', age = 50, SkinDisease.medication.relf = 'A'), type = "response")

cat(relief_probit.pred)
```
$P(relief)^0 = \phi ⋅ (1.92301 ⋅ (male) + 0.10260 ⋅ (age) + 2.23351 ⋅ (MedicationA) - 4.07825)$
$P(relief)^0 = \phi ⋅ (1.92301 ⋅ (0) + 0.10260 ⋅ (50) + 2.23351 ⋅ (1) - 4.07825)$
$P(relief)^0 = \phi ⋅ (3.28526)$
$P(relief)^0 = 0.9994$

_Cell Output:_
```
	Probit Model: Prediction
0.9994907
```
* Complementary Log-Log Model
```R
# Probit Model
# checking model fit
cat('\tCLogLog Model: Prediction\n')
relief_cloglog.pred <- predict(
  relief_cloglog.fitted.model,
  data.frame(SkinDisease.gender.relf = 'F', age = 50, SkinDisease.medication.relf = 'A'),
  type = "response"
)

cat(relief_probit.pred)
```
$P(relief)^0 = 1 - e^{-e^{1.92301 ⋅ (male) + 0.10260 ⋅ (age) + 2.23351 ⋅ (MedicationA) - 4.07825}}$</br>
$P(relief)^0 = 1 - e^{-e^{1.92301 ⋅ (0) + 0.10260 ⋅ (50) + 2.23351 ⋅ (1) - 4.07825}}$ </br>
$P(relief)^0 = 1 - e^{-e^{3.26526}}$</br>
$P(relief)^0 = 1$

_Cell Output:_
```
	CLogLog Model: Prediction
0.9994907
```

# Example 2
A bank needs to estimate the default rate of customers' home equity loans. A random sample of 35 customers is drawn. The selected variables are the loan-to-value (LTV) ratio defined as the ratio of a loan to the value of an asset purchased (in percent), age (in years), income (high/low), and response (yes=default, no=payoff).

**Read CSV file into Colab**
```R
QualityLoans.data <- read_csv('https://raw.githubusercontent.com/coronasanchez/LACC-MATH-239/main/HW3/Exercise3.4Data.csv')

attach(QualityLoans.data)
head(QualityLoans.data)
```
**Fit the Model**
```R
# Rescaling Variables and specifying reference categories
QualityLoans.income.relf <- relevel(factor(QualityLoans.data$income, order = FALSE), ref = 'high')
QualityLoans.default.relf <- relevel(factor(QualityLoans.data$default, order = FALSE), ref = 'no')
```
* Binary Logistic Regression Model
```R
# Binary Logistic Regression Model
# Fit the generalized linear model to transformed response
default_bi.fitted.model <- glm (
  QualityLoans.default.relf ~ LTV + age + QualityLoans.income.relf,
  data = QualityLoans.data,
  family = binomial(link = logit)
)

# Print the summary of the model
summary(default_bi.fitted.model)
```
_Cell Output:_
```
 
Call:
glm(formula = QualityLoans.default.relf ~ LTV + age + QualityLoans.income.relf, 
    family = binomial(link = logit), data = QualityLoans.data)

Coefficients:
                            Estimate Std. Error z value Pr(>|z|)  
(Intercept)                 -3.00869    4.09545  -0.735   0.4626  
LTV                          0.10586    0.05124   2.066   0.0388 *
age                         -0.16157    0.07314  -2.209   0.0272 *
QualityLoans.income.relflow  1.11619    1.02490   1.089   0.2761  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 45.004  on 34  degrees of freedom
Residual deviance: 28.469  on 31  degrees of freedom
AIC: 36.469

Number of Fisher Scoring iterations: 6
```
**Reduce Binary Fitted Model:** $\frac{\widehat{\pi} (DefaultRate)}{1-\widehat{\pi}(DefaultRate)}$ $=e^{0.10586 ⋅ (LTV) - 0.16157 ⋅ (age)}$

Extracting AIC, AICC, and BIC. 
```R
# number of coeffient
default_bi.numcoeff <- length(coef(default_bi.fitted.model))
cat("p = ", default_bi.numcoeff)

# Binary Logistic Regression Model
# extracting AICC and BIC for fitted model
cat("\tBinary Logistic\n")
cat("AIC  = \t", AIC(default_bi.fitted.model))

# -2logLik(fitted.model)+2*p*n / (n- p -1)
default_bi.AICC <- -2 * logLik(default_bi.fitted.model) + 2 * default_bi.numcoeff * QualityLoans.n / (QualityLoans.n - default_bi.numcoeff - 1)
cat('\nAICC = \t', default_bi.AICC)

default_bi.BIC <- BIC(default_bi.fitted.model)
cat("\nBIC  = \t", default_bi.BIC)
```
_Cell Output:_
```
p =  4

	Binary Logistic
AIC  = 	 36.46933
AICC = 	 37.80266
BIC  = 	 42.69072
```

* Probit Model
```R
# Probit Model
# Fit the generalized linear model to transformed response
default_probit.fitted.model <- glm(
  QualityLoans.default.relf ~ LTV + age + QualityLoans.income.relf,
  data = QualityLoans.data,
  family = binomial(link = probit)
  )

# Print the summary of the model
summary(default_probit.fitted.model)

# outputting estimated sigma
# cat("\nsigma:\t", sigma(bmi.fitted.model)
```

_Cell Output:_
```R
 
Call:
glm(formula = QualityLoans.default.relf ~ LTV + age + QualityLoans.income.relf, 
    family = binomial(link = probit), data = QualityLoans.data)

Coefficients:
                            Estimate Std. Error z value Pr(>|z|)  
(Intercept)                 -1.60587    2.34383  -0.685   0.4933  
LTV                          0.06200    0.02853   2.173   0.0297 *
age                         -0.09865    0.04121  -2.394   0.0167 *
QualityLoans.income.relflow  0.63924    0.59365   1.077   0.2816  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 45.004  on 34  degrees of freedom
Residual deviance: 28.173  on 31  degrees of freedom
AIC: 36.173

Number of Fisher Scoring iterations: 7
```
**Reduce Probit fitted model:** $\phi^{-1} ⋅ \widehat{\pi}(DefaultRate)$ $=0.06200 ⋅ (LTV) - 0.09865 ⋅ (age)$

Extracting AIC, AICC, and BIC. 
```R
# Probit Model
# number of coeffient
default_probit.numcoeff <- length(coef(default_probit.fitted.model))
cat("p = ", default_probit.numcoeff)

cat('\n\tProbit Model\n')
# extracting AICC and BIC for fitted model
cat('AIC  =\t', AIC(default_probit.fitted.model))
default_probit.AICC <- -2 * logLik(default_probit.fitted.model) + 2 * default_probit.numcoeff * QualityLoans.n / (QualityLoans.n - default_probit.numcoeff - 1)
cat('\nAICC =\t', default_probit.AICC)
cat('\nBIC  =\t', BIC(default_probit.fitted.model))
```
_Cell Output:_
```
p =  4
	Probit Model
AIC  =	 36.17332
AICC =	 37.50665
BIC  =	 42.39471
```
* Complementary Log-Log Model
```R
# Complementary Log-Log Model
# Fitting complementary log-log model
default_cll.fitted.model <- glm(
  QualityLoans.default.relf ~ LTV + age + QualityLoans.income.relf,
  data = QualityLoans.data,
  family = binomial(link = cloglog)
)
# Print the summary of the model
summary(default_cll.fitted.model)
```
_Cell Output:_
```
 
Call:
glm(formula = QualityLoans.default.relf ~ LTV + age + QualityLoans.income.relf, 
    family = binomial(link = cloglog), data = QualityLoans.data)

Coefficients:
                            Estimate Std. Error z value Pr(>|z|)  
(Intercept)                 -2.68156    3.27613  -0.819   0.4131  
LTV                          0.07896    0.04114   1.919   0.0550 .
age                         -0.12254    0.05111  -2.398   0.0165 *
QualityLoans.income.relflow  0.89073    0.72016   1.237   0.2161  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 45.004  on 34  degrees of freedom
Residual deviance: 28.436  on 31  degrees of freedom
AIC: 36.436

Number of Fisher Scoring iterations: 7
```
**Complementary Log-Log Fitted Model:** $1 - \widehat{\pi}(DefaultRate) = e^{-e^{0.07896 ⋅ (LTV) - 0.12254 ⋅ (age) + 0.89073 ⋅ (IncomeLow) - 2.68156}}$

Extracting AIC, AICC, and BIC. 
```R
# Complementary Log-Log Model
# number of coeffient
default_cll.numcoeff <- length(coef(default_cll.fitted.model))
cat("p = ", default_cll.numcoeff)

cat('\n\tCLog Log\n')
# extracting AICC and BIC for fitted model
cat('AIC  =\t', AIC(default_cll.fitted.model))
default_cll.AICC <- -2 * logLik(default_cll.fitted.model) + 2 * default_cll.numcoeff * QualityLoans.n / ( QualityLoans.n - default_cll.numcoeff - 1)
cat('\nAICC =\t', default_cll.AICC)
cat('\nBIC  =\t', BIC(default_cll.fitted.model))
```
_Cell Output:_
```
p =  4
	CLog Log
AIC  =	 36.43576
AICC =	 37.7691
BIC  =	 42.65716
```
**Model Fits the data well?**
* Binary Logistic Regression Model
```R
# Binary Logistic Regression Model
# Checking model fit
cat("\tBinary Logistic\n")
default_bi.null.model <- glm(QualityLoans.default.relf ~ 1, data = QualityLoans.data, family = binomial(link=logit))

default_bi.deviance <- -2*(logLik(default_bi.null.model)-logLik(default_bi.fitted.model))

cat("deviance =\t", default_bi.deviance)

default_bi.pvalue <- pchisq(default_bi.deviance, df = default_bi.numcoeff - 1, lower.tail = FALSE)
cat("\np-value =\t", default_bi.pvalue)
```
_Cell Output:_
```
	Binary Logistic
deviance =	 16.53454
p-value =	 0.0008808876
```
* Probit Model
```R
# Probit Model
# checking model fit
cat('\tProbit Model\n')
default_probit.null.model <- glm(
  QualityLoans.default.relf ~ 1,
  data = QualityLoans.data,
  family = binomial(link = probit)
)

default_probit.deviance <- -2*(logLik(default_probit.null.model)-logLik(default_probit.fitted.model))
cat("deviance =", default_probit.deviance)

default_pvalue <- pchisq(default_probit.deviance, df = default_probit.numcoeff - 1, lower.tail = FALSE)
cat("\np-value =", default_pvalue)
```
_Cell Output:_
```
	Probit Model
deviance = 16.83055
p-value = 0.0007657748
```
* Complementary Log-Log Model
```R
# CLogLog Model
#checking model fit
cat('\tClog Log Model\n')
default_cll.null.model <- glm(QualityLoans.default.relf ~ 1,data = QualityLoans.data, family = binomial(link = cloglog))
default_cll.deviance <- -2 * (logLik(default_cll.null.model) - logLik(default_cll.fitted.model))
cat("deviance = ", default_cll.deviance)

default_cll.pvalue <- pchisq(default_cll.deviance, df = default_cll.numcoeff - 1, lower.tail = FALSE)
cat("\np-value = ", default_cll.pvalue)
```
_Cell Output:_
```
	Clog Log Model
deviance =  16.56811
p-value =  0.0008670163
```
**Using the fitted model to predict.**
Give a point estimate for the probability of loan default if the LTV ratio is 50%, and the borrower is a 50-year-old man with high income.
* Binary Logistic Regression Model
```R
# Binary Logistic Regression Model
# Using fitted moel for predition
cat('\tBinary Model: Prediction\n')
QualityLoans_bi.predict <- predict(
  default_bi.fitted.model,
  data.frame(
    LTV = 50, age = 50 , QualityLoans.income.relf = 'high'), type = "response")

cat(QualityLoans_bi.predict)
check_default(QualityLoans_bi.predict)
```
$P(DefaultRate)^0 = \frac{e^{0.10586 ⋅ (LTV) - 0.16157 ⋅ (age) + 1.11619 ⋅ (IncomeLow) - 3.00869}}{1+e^{0.10586 ⋅ (LTV) - 0.16157 ⋅ (age) + 1.11619 ⋅ (IncomeLow) - 3.00869}}$
$P(DefaultRate)^0 = \frac{e^{0.10586 ⋅ (50) - 0.16157 ⋅ (50) + 1.11619 ⋅ (0) - 3.00869}}{1+e^{0.10586 ⋅ (50) - 0.16157 ⋅ (50) + 1.11619 ⋅ (0) - 3.00869}}$
$P(DefaultRate)^0 = \frac{e^{-5.79419}}{1-e^{-5.79419}}$
$P(DefaultRate)^0 = 0.00305$

_Cell Output:_
```
	Binary Model: Prediction
0.00303576	[no]
```
* Probit Model
```R
# Probit Model
# checking model fit
cat('\tProbit Model: Prediction\n')
QualityLoans_probit.pred <- predict(
  default_probit.fitted.model,
  data.frame(LTV = 50, age = 50, QualityLoans.income.relf = 'high'), type = "response")

cat(QualityLoans_probit.pred)
check_default(QualityLoans_probit.pred)
```
$P(DefaultRate)^0 = \phi ⋅ 0.06200 ⋅ (LTV) - 0.09865 ⋅ (age) + 0.63924 ⋅ (IncomeLow) - 1.60587$
$P(relief)^0 = \phi ⋅ 0.06200 ⋅ (50) - 0.09865 ⋅ (50) + 0.63924 ⋅ (0) - 1.60587$
$P(relief)^0 = \phi ⋅ (-3.43837)$
$P(relief)^0 = 0.00029$

_Cell Output:_
```
	Probit Model: Prediction
0.0002923167	[no]
```
* Complementary Log-Log Model
```R
# Probit Model
# checking model fit
cat('\tCLogLog Model: Prediction\n')
QualityLoans_cll.pred <- predict(
  default_cll.fitted.model,
  # data = QualityLoans.data,
  data.frame(LTV = 50 , age = 50, QualityLoans.income.relf = 'high'), type = "response")
cat(QualityLoans_cll.pred)
cat(check_default(QualityLoans_cll.pred))
```
$1 - \widehat{\pi}(DefaultRate) = e^{-e^{0.07896 ⋅ (50) - 0.12254 ⋅ (50) + 0.89073 ⋅ (0) - 2.68156}}$
$P(relief)^0 = 1 - e^{-e^{0.07896 ⋅ (50) - 0.12254 ⋅ (50) + 0.89073 ⋅ (0) - 2.68156}}$
$P(relief)^0 = 1 - e^{-e^{-4.86056}}$
$P(relief)^0 = 0.0077$

_Cell Output:_
```
	CLogLog Model: Prediction
0.007713725	[no]
```

