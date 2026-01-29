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
Binary Fitted Model: $\frac{\widehat{\pi} (relief)}{1-\widehat{\pi}(relief)}$ $=e^{3.17132 ⋅ (male) + 0.17131 ⋅ (age) + 3.81641 ⋅ (MedicationA) - 6.79921}$  

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
Probit Fitted Model: $\phi^{-1} ⋅ \widehat{\pi}(relief)$ $=1.92301 ⋅ (male) + 0.10260 ⋅ (age) + 2.23351 ⋅ (MedicationA) - 4.07825$

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
Complementary Log-Log Fitted Model: $1 - \widehat{\pi}(relief) = e^{-e^{1.92301 ⋅ (male) + 0.10260 ⋅ (age) + 2.23351 ⋅ (MedicationA) - 4.07825}}$

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
