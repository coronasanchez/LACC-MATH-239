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
* Cumulative Logit Model
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

$\hat P(\text{admit}) / (1 - \hat P(\text{admit}))
= e^{3.119391 \cdot \text{GPA} + 0.027755 \cdot \text{GMAT} - 23.390}$



$\frac{\widehat{\mathbb{P}}(\text{admit, or border})}{\widehat{\mathbb{P}}(\text{not admit})} = e^{3.119391 ⋅ (GPA) + 0.027755 ⋅ (GMAT) - 21.853}$

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
Predict the probabilities of each admission status for a person whose GPA is 3.1 and GMAT score is 550.

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
