# Regression Models for Categorical Response
* Cumulative Logit Model
* Cumulative Probit Model
* Cummulative Complementary Log-Log Model
* Generalized Logit Model for Nominal Response

**Before starting, run the code below to import all necessary functions and libraries.**
```R
install.packages("ordinal")
```
```R
library(readr)
library(ggplot2)
library(ordinal)
```

## Example 1
Grade point average (GPA) and graduate management aptitude test (GMAT) scores are used by the admission office of a business school to decide which applicants should be admitted to the school's graduate program. The data below are GPA and GMAT scores for 42 recent applicants who have been categorized as admitted, borderline, or not admitted.

**Read CSV file into Colab.**
```R
# Use 'raw.githubusercontent.com' to access raw file content
record.data <- read_csv('https://raw.githubusercontent.com/coronasanchez/LACC-MATH-239/main/HW4/Exercise4.1Data.csv')
head(record.data)

attach(record.data)
```
```R
# Rescaling Variables and specifying reference categories
record_status.rel <- relevel(factor(record.data$status, order = FALSE), ref = "admit")
```
### Cumulative Logit Model
<p align = "center"> Fit the model </p>

```R
# fitting cumulative logit model
record_clm.fitted.model <- clm(
  record_status.rel ~ GPA + GMAT,
  data = record.data,
  link = "logit"
)

summary(record_clm.fitted.model)
```
_Cell Output:_
```
 Warning message:
“(3) Model is nearly unidentifiable: large eigenvalue ratio
 - Rescale variables? 
In addition: Absolute and relative convergence criteria were met”
formula: record_status.rel ~ GPA + GMAT
data:    record.data

 link  threshold nobs logLik AIC   niter max.grad cond.H 
 logit flexible  42   -27.54 63.09 6(0)  7.39e-13 1.1e+08

Coefficients:
      Estimate Std. Error z value Pr(>|z|)    
GPA  -3.119391   1.191309  -2.618 0.008833 ** 
GMAT -0.027755   0.008406  -3.302 0.000961 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Threshold coefficients:
                Estimate Std. Error z value
admit|border     -23.390      5.857  -3.994
border|notadmit  -21.853      5.670  -3.854
```
**Fitted Cumulative Logit Model:**

$\quad\quad\frac{\hat P(\text{admit})}{(1 - \hat P(\text{admit}))} = e^{3.119391 \cdot \text{GPA} + 0.027755 \cdot \text{GMAT} - 23.390}$

$\quad\quad\frac{\widehat{\mathbb{P}}(\text{admit, or border})}{\widehat{\mathbb{P}}(\text{not admit})} = e^{3.119391 ⋅ (GPA) + 0.027755 ⋅ (GMAT) - 21.853}$

Extracting AIC, AICC, and BIC. 
```R
# extracting AICC and BIC for fitted model
record_clm.p <- length(coefficients(record_clm.fitted.model))
cat("p =", record_clm.p)

record_clm.n <- nobs(record_clm.fitted.model)
cat("\nn =", record_clm.n)

cat("\n\nAIC  :", AIC(record_clm.fitted.model))
record_clm.AICC = -2 * logLik(record_clm.fitted.model) + 2 * record_clm.p * (record_clm.n/(record_clm.n - record_clm.p - 1))
cat("\nAICC :", record_clm.AICC)
cat("\nBIC  :", BIC(record_clm.fitted.model))
```
```
p = 4
n = 42

AIC  : 63.08868
AICC : 64.16976
BIC  : 70.03935
```
<p align = "center"> Check Model Fit </p>

```R
# Checking model fit
record_clm.null.model <- clm(
  record_status.rel ~ 1,
  data = record.data,
  link = "logit"
)

record_clm.deviance <- -2*(logLik(record_clm.null.model) - logLik(record_clm.fitted.model))
print(record_clm.deviance)

record_clm.pvalue <- pchisq(
  record_clm.deviance,
  df = length(record_clm.fitted.model$beta),
  lower.tail = FALSE
)
print(record_clm.pvalue)
```
_Cell Output:_
```
'log Lik.' 31.04588 (df=2)
'log Lik.' 1.813312e-07 (df=2)
```
<p align = "Center"> Predict Model Fit </p>

> Predict the probabilities of each admission status for a person whose GPA is 3.1 and GMAT score is 550. We calculate:

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = \frac{e^{3.119391 ⋅ (GPA) + 0.027755 ⋅ (GMAT) - 23.390}}{1+e^{3.119391 ⋅ (GPA) + 0.027755 ⋅ (GMAT) - 23.390}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = \frac{e^{3.119391 ⋅ (3.1) + 0.027755 ⋅ (550) - 23.390}}{1+e^{3.119391 ⋅ (3.1) + 0.027755 ⋅ (550) - 23.390}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = \frac{e^{1.5453621}}{1+e^{1.5453621}} = 0.8242$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})}= \frac{e^{3.119391 ⋅ (GPA) + 0.027755 ⋅ (GMAT) - 21.853}}{1+e^{3.119391 ⋅ (GPA) + 0.027755 ⋅ (GMAT) - 21.853}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})}= \frac{e^{3.119391 ⋅ (3.1) + 0.027755 ⋅ (550) - 21.853}}{1+e^{3.119391 ⋅ (3.1) + 0.027755 ⋅ (550) - 21.853}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})}= \frac{e^{3.0823621}}{1+e^{3.0823621}} = 0.9562$

_The predicted probabilities for individual categories are:_

$\quad\quad\hat P(\text{border}) = \hat P(\text{admit or border}) - \hat P(\text{admit})$

$\quad\quad{\widehat{\mathbb{P}}(\text{border})} = 0.9562 - 0.8242 = 0.132$

$\quad\quad{\widehat{\mathbb{P}}(\text{not admit})}= 1 - {\widehat{\mathbb{P}}(\text{admit, or border})}$

$\quad\quad{\widehat{\mathbb{P}}(\text{not admit})} = 1 - 0.9562 = 0.0438$</center>

_Check:_

$\quad\quad\hat P(\text{admit}) + \hat P(\text{border}) + \hat P(\text{not admit}) = 1$

$\quad\quad0.8242 + 0.132 + 0.0438 = 1$

```R
# using fitted model for prediction
record_clm.pred <- predict(
  record_clm.fitted.model,
  data.frame(
    GPA = 3.1,
    GMAT = 550,
    type = "prob"
  )
)
print(record_clm.pred)

print(sum(unlist(record_clm.pred)))
```

_Cell Output:_
```
$fit
      admit    border   notadmit
1 0.8241952 0.1319818 0.04382299

[1] 1
```
### Cumulative Probit Model
<p align = "center"> Fit the model </p>

```R
# Fitting cumulative probit model after scaling
record_cpm.fitted.model <- clm(
  record_status.rel ~ GPA + GMAT,
  data = record.data,
  link = "probit"
)

summary(record_cpm.fitted.model)
```
_Cell Output:_
```
 Warning message:
“(3) Model is nearly unidentifiable: large eigenvalue ratio
 - Rescale variables? 
In addition: Absolute and relative convergence criteria were met”
formula: record_status.rel ~ GPA + GMAT
data:    record.data

 link   threshold nobs logLik AIC   niter max.grad cond.H 
 probit flexible  42   -27.59 63.19 6(0)  2.00e-11 9.5e+07

Coefficients:
      Estimate Std. Error z value Pr(>|z|)    
GPA  -1.735613   0.622156  -2.790 0.005276 ** 
GMAT -0.016546   0.004573  -3.618 0.000297 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Threshold coefficients:
                Estimate Std. Error z value
admit|border     -13.603      3.078   -4.42
border|notadmit  -12.724      3.001   -4.24
```
Fitted Cumulative Probit Model form:

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = Φ ⋅ (1.735613 ⋅ (GPA) + 0.016546 ⋅ (GMAT) - 13.603)$
$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})} = Φ ⋅ (1.735613 ⋅ (GPA) + 0.016546 ⋅ (GMAT) - 12.724)$

Exctacting AIC, AICC, and BIC. 
```R
# extracting AICC and BIC for fitted model
record_cpm.p <- length(coefficients(record_cpm.fitted.model))
cat("p =", record_cpm.p)

record_cpm.n <- nobs(record_cpm.fitted.model)
cat("\nn =", record_cpm.n)

cat("\n\nAIC  :", AIC(record_cpm.fitted.model))
record_cpm.AICC = -2 * logLik(record_cpm.fitted.model) + 2 * record_cpm.p * (record_cpm.n/(record_cpm.n - record_cpm.p - 1))
cat("\nAICC :", record_cpm.AICC)
cat("\nBIC  :", BIC(record_cpm.fitted.model))
```
_Cell Output:_
```
p = 4
n = 42

AIC  : 63.18598
AICC : 64.26707
BIC  : 70.13666
```
<p align = "center"> Check Model Fit </p>

```R
# Checking model fit
record_cpm.null.model <- clm(
  record_status.rel ~ 1,
  data = record.data,
  link = "probit"
)

record_cpm.deviance <- -2 * (logLik(record_cpm.null.model) - logLik(record_cpm.fitted.model))
print(record_cpm.deviance)

record_cpm.pvalue <- pchisq(
  record_cpm.deviance,
  df = length(record_cpm.fitted.model$beta),
  lower.tail = FALSE
)

print(record_cpm.pvalue)
```
_Cell Output:_
```
'log Lik.' 30.94857 (df=2)
'log Lik.' 1.903718e-07 (df=2)
```
<p align = "Center"> Predict Model Fit </p>

> Suppose we would like to find the predicted probability of each admission status for a person whose GPA is 3.1 and GMAT score is 550. We calculate:

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = Φ ⋅ (1.735613 ⋅ (GPA) + 0.016546 ⋅ (GMAT) - 13.603)$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = Φ ⋅ (1.735613 ⋅ (3.1) + 0.016546 ⋅ (550) - 13.603)$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = Φ ⋅ (0.877700300000003)$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = 0.8099$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})} = Φ ⋅ (1.735613 ⋅ (GPA) + 0.016546 ⋅ (GMAT) - 12.724)$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})} = Φ ⋅ (1.735613 ⋅ (3.1) + 0.016546 ⋅ (550) - 12.724)$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})} = Φ ⋅ (1.7567003)$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})} = 0.9605$

_The predicted probabilities for individual categories are:_

$\quad\quad{\widehat{\mathbb{P}}(\text{border})} = {\widehat{\mathbb{P}}(\text{admit, or border})} - {\widehat{\mathbb{P}}(\text{admit})}$

$\quad\quad{\widehat{\mathbb{P}}(\text{border})} = 0.9605 - 0.8099$

$\quad\quad{\widehat{\mathbb{P}}(\text{border})} = 0.1506$

$\quad\quad{\widehat{\mathbb{P}}(\text{not admit})} = 1 - {\widehat{\mathbb{P}}(\text{admit, or border})}$

$\quad\quad{\widehat{\mathbb{P}}(\text{not admit})} = 1 - 0.9605 = 0.0395$

_Check:_

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} + {\widehat{\mathbb{P}}(\text{border})} + {\widehat{\mathbb{P}}(\text{not admit})} = 1$

$\quad\quad0.8099 + 0.1506 + 0.0395 = 1$</center>

```R
# using fitted model for prediction
record_cpm.pred <- predict(
  record_cpm.fitted.model,
  data.frame(
    GPA = 3.1,
    GMAT = 550,
    type = "prob"
  )
)
print(record_cpm.pred)

print(sum(unlist(record_cpm.pred)))
```
_Cell Output:_
```
$fit
      admit    border   notadmit
1 0.8098528 0.1506302 0.03951702

[1] 1
```

### Cumulative Complementary Log-Log Model
<p align = "center"> Fit the model </p>

```R
# Fitting cumulative complementary log-log model
record_ccll.fitted.model <- clm(
  record_status.rel ~ GPA + GMAT,
  data = record.data,
  link = "cloglog"
)

summary(record_ccll.fitted.model)
```
_Cell Output:_
```
 Warning message:
“(3) Model is nearly unidentifiable: large eigenvalue ratio
 - Rescale variables? 
In addition: Absolute and relative convergence criteria were met”
formula: record_status.rel ~ GPA + GMAT
data:    record.data

 link    threshold nobs logLik AIC   niter max.grad cond.H 
 cloglog flexible  42   -29.50 66.99 6(0)  2.27e-13 9.6e+07

Coefficients:
      Estimate Std. Error z value Pr(>|z|)    
GPA  -1.647883   0.631617  -2.609 0.009081 ** 
GMAT -0.015190   0.004303  -3.530 0.000415 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Threshold coefficients:
                Estimate Std. Error z value
admit|border     -13.137      3.059  -4.294
border|notadmit  -12.265      2.989  -4.104
```
Fitted Cumulative Complementary Log-Log Model form:

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = 1 - e^{-e^{1.647883 ⋅ (GPA) + 0.015190 ⋅ (GMAT) - 13.137}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})} = 1 - e^{-e^{1.647883 ⋅ (GPA) + 0.015190 ⋅ (GMAT) - 12.265}}$

Extracting AIC, AICC, and BIC. 
```R
# extracting AICC and BIC for fitted model
record_ccll.p <- length(coefficients(record_ccll.fitted.model))
cat("p =", record_ccll.p)

record_ccll.n <- nobs(record_ccll.fitted.model)
cat("\nn =", record_ccll.n)

cat("\n\nAIC  :", AIC(record_ccll.fitted.model))
record_ccll.AICC = -2 * logLik(record_ccll.fitted.model) + 2 * record_ccll.p * (record_ccll.n/(record_ccll.n - record_ccll.p - 1))
cat("\nAICC :", record_ccll.AICC)
cat("\nBIC  :", BIC(record_ccll.fitted.model))
```
_Cell Output:_
```
p = 4
n = 42

AIC  : 66.99011
AICC : 68.07119
BIC  : 73.94079
```

<p align = "center"> Check Model Fit </p>

```R
# Checking model fit
record_ccll.null.model <- clm(
  record_status.rel ~ 1,
  data = record.data,
  link = "cloglog"
)

record_ccll.deviance <- -2 * (logLik(record_ccll.null.model) - logLik(record_ccll.fitted.model))
print(record_ccll.deviance)

record_ccll.pvalue <- pchisq(
  record_ccll.deviance,
  df = length(record_ccll.fitted.model$beta),
  lower.tail = FALSE
)

print(record_ccll.pvalue)
```
_Cell Output:_
```
'log Lik.' 27.14445 (df=2)
'log Lik.' 1.275435e-06 (df=2)
```

<p align = "Center"> Predict Model Fit </p>

> Suppose we would like to find the predicted probability of each admission status for a person whose GPA is 3.1 and GMAT score is 550. We calculate:

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = 1 - e^{-e^{1.647883 ⋅ (GPA) + 0.015190 ⋅ (GMAT) - 13.137}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = 1 - e^{-e^{1.647883 ⋅ (3.1) + 0.015190 ⋅ (550) - 13.137}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = 1 - e^{-e^{0.3259373}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} = 0.7498$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})} = 1 - e^{-e^{1.647883 ⋅ (GPA) + 0.015190 ⋅ (GMAT) - 12.265}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})} = 1 - e^{-e^{1.647883 ⋅ (3.1) + 0.015190 ⋅ (550) - 12.265}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})} = 1 - e^{-e^{1.1979373}}$

$\quad\quad{\widehat{\mathbb{P}}(\text{admit, or border})} = 0.9636$</center>

_The predicted probabilities for individual categories are:_

$\quad\quad{\widehat{\mathbb{P}}(\text{border})} = {\widehat{\mathbb{P}}(\text{admit, or border})} - {\widehat{\mathbb{P}}(\text{admit})}$

$\quad\quad{\widehat{\mathbb{P}}(\text{border})} = 0.9636 - 0.7498 = 0.2138$

$\quad\quad{\widehat{\mathbb{P}}(\text{not admit})} = 1 - {\widehat{\mathbb{P}}(\text{admit, or border})}$

$\quad\quad{\widehat{\mathbb{P}}(\text{not admit})} = 1 - 0.9636 = 0.0364$</center>

_Check:_

$\quad\quad{\widehat{\mathbb{P}}(\text{admit})} + {\widehat{\mathbb{P}}(\text{border})} + {\widehat{\mathbb{P}}(\text{not admit})} = 1$

$\quad\quad0.7498 + 0.2138 + 0.0364 = 1$

```R
# using fitted model for prediction
record_ccll.pred <- predict(
  record_ccll.fitted.model,
  data.frame(
    GPA = 3.1,
    GMAT = 550,
    type = "prob"
  )
)
print(record_ccll.pred)
print(sum(unlist(record_ccll.pred)))
```
_Cell Output:_
```
$fit
      admit    border   notadmit
1 0.7497578 0.2138205 0.03642168

[1] 1
```
