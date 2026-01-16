# Regression Models for Response with Right-skewed Distribution
We learn and utilize: 
* Box-Cox Power Transformation
* Gamma Regression Model

The solutions are provided to run in Colab, make sure that your runtype runs *R*. If you wish to run the code in R studio, the code will work well.</br> 
<p align="center">
  <img width="350" height="auto" alt="image" src="https://github.com/user-attachments/assets/e10980da-4529-47b4-8a1e-7180e948b0f4" />
</p>

If using Colab, we must always install the packages, unlike, R studio, the package and be install and selected once. 
```R
install.packages('MASS')
```

Then we must read the libraries. 
```R
library(readr)
library(ggplot2)
library(MASS)
```

## Example 1
An intervention-control study on childhood obesity was conducted at a children's clinic. A cohort of 36 obese children, ages 6 through 16, were followed for 9 months. The intervention consisted of educational sessions for parents and vigorous exercise activities for kids. The control group participants were provided with resources regarding other active and healthy lifestyle programs offered in their community. Their gender(M/F), age (in years), group (intervention Tx or control Cx), and percentiles for pre-BMI and post-BMI were recorded. 

**Read CSV file into Colab.**
```R
# Read .cvs file into Colab in R code from GitHub
bmi.data <- read_csv('https://raw.githubusercontent.com/coronasanchez/LACC-MATH-239/main/HW2/Exercise2.1Data.csv')

attach(bmi.data)
head(bmi.data)
```

**Plot the histogram for the scores (y) and conduct the normality tests.**
```R
bmi_diff <- preBMI - postBMI

bmi_hist <- ggplot(bmi.data, aes(x = bmi_diff)) +
  geom_histogram(aes(y = after_stat(density)), bins = 8, fill ="gray",color = "black") +
    stat_function(fun = dnorm, args = list(mean = mean(bmi_diff), sd = sd(bmi_diff)), color = "red", linewidth = 1) +
      labs(title = "The difference in BMI with Normal Distribution Line with fitted line", x = "Diff BMI", y = "Relative Frequency") +
        theme_bw()

print(bmi_hist)

shapiro.test(bmi_diff)
```
_Cell output:_
```
 
	Shapiro-Wilk normality test

data:  bmi_diff
W = 0.79159, p-value = 1.114e-05
```
<p align="center">
  <img width="400" height="auto" alt="image" src="https://github.com/user-attachments/assets/13ec3bb6-a283-4c89-8403-8727fda0767a" />
</p>  

We must rescale
```R
# Rescaling Variables and specifying reference categories
bmi_gender.relf <- relevel(factor(bmi.data$gender, order = FALSE), ref = 'F')
bmi_group.relf <- relevel(factor(bmi.data$group, order = FALSE), ref = 'Cx')

# uncomment to view how the column was leveled.
# print(unclass(gender.relf))
# print(unclass(group.relf))

# Finding optimal lambda for Box-Cox transformation
bmi.bxcx.fit <- boxcox(bmi_diff ~ bmi_gender.relf + age + bmi_group.relf, data = bmi.data, lambda = seq(-3,3,1/4), interp = FALSE)
bmi.bxcx.data <- data.frame(bmi.bxcx.fit$x, bmi.bxcx.fit$y)
bmi.ordered.data <- bmi.bxcx.data[with(bmi.bxcx.data, order(-bmi.bxcx.fit.y)),]

# Display
bmi.ordered.data[1,]
```

Then we can apply a box-cox tranformation. 
```R
# applying box-cox transformation with lambda = 0
bmi_diff.tr <- log(bmi_diff)

# plotting histogram for tranformed response
bmi_hist.tr <- ggplot(bmi.data, aes(x = bmi_diff.tr)) +
  geom_histogram(aes(y = after_stat(density)), bins = 8, fill = "lightblue", color = "black") +
    stat_function(fun = dnorm, args = list(mean = mean(bmi_diff.tr), sd = sd(bmi_diff.tr)), color = "red", linewidth = 1) +
      labs(title = "BMI with Normal Distribution Line after Box-Cox Transformation", x = "Diff BMI", y = "Relative Frequency") +
        theme_bw()

print(bmi_hist.tr)

shapiro.test(bmi_diff.tr)
```
_Cell output:_
```
 
	Shapiro-Wilk normality test

data:  bmi_diff.tr
W = 0.9877, p-value = 0.9532
```
<p align="center">
  <img width="400" height="auto" alt="image" src="https://github.com/user-attachments/assets/065d2c77-f308-4553-976c-73a6554226de" />
</p>

**Fit a general linear model.**
```R
# Fit the generalized linear model to transformed response
bmi.fitted.model <- glm(bmi_diff.tr ~ bmi_gender.relf + age + bmi_group.relf, data = bmi.data, family = gaussian(link = identity))

# Print the summary of the model
summary(bmi.fitted.model)

# outputting estimated sigma
cat("\nsigma:\t", sigma(bmi.fitted.model))
```
_Cell output:_
```
 
Call:
glm(formula = bmi_diff.tr ~ bmi_gender.relf + age + bmi_group.relf, 
    family = gaussian(link = identity), data = bmi.data)

Coefficients:
                 Estimate Std. Error t value Pr(>|t|)    
(Intercept)      -0.29322    0.43014  -0.682 0.500346    
bmi_gender.relfM  0.49862    0.21708   2.297 0.028317 *  
age               0.05008    0.03651   1.372 0.179731    
bmi_group.relfTx  0.93835    0.22307   4.207 0.000195 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 0.4187965)

    Null deviance: 25.432  on 35  degrees of freedom
Residual deviance: 13.401  on 32  degrees of freedom
AIC: 76.59

Number of Fisher Scoring iterations: 2

sigma:	 0.6471448
```
**Reduced Fitted Model** $\widehat{\mathbb{E}}(ln(BMI)) = 0.49862 ⋅ (Male) + 0.93835 ⋅ (GroupTx)$

**Model Fits the data well?**
```R
# checking model fit
bmi_null.model <-
  glm(bmi_diff.tr ~ 1, data = bmi.data, family = gaussian(link = identity))

# Get total num of coeff
num_coeff_bmi_df <- length(coef(bmi.fitted.model)) - 1
cat("total num of DF:", num_coeff_bmi_df, "\n")

deviance <- -2*(logLik(bmi_null.model)-logLik(bmi.fitted.model))
cat("deviance =\t", deviance,"\n")

p.value <- pchisq(deviance, df = num_coeff_bmi_df, lower.tail = FALSE)
cat("p-value =\t", p.value, "\n")
```
_Cell output:_ 
```
total num of DF: 3 
deviance =	 23.06361 
p-value =	 3.916872e-05 
```
Predict the change in BMI percentile for a 9-year-old girl in the control group.
</br>
$BMI^0 = exp^{(-0.29322 + 0.49862 ⋅ (0) + 0.05008 ⋅ (9) + 0.93835 ⋅ (0))}$ <br>
$BMI^0 = 1.1705$

```R
bmi_predict.tr <- predict(bmi.fitted.model, data.frame(bmi_gender.relf  = "F", age = 9, bmi_group.relf = "Cx"))

# Overwrite
bmi_predict.tr <- exp(bmi_predict.tr)
cat("BMI:", bmi_predict.tr )
```
_Cell Output:_
```
BMI: 1.170609
```

## Example 2
Investigators at a large medical center conducted a quality improvement (QI) study which consisted of a six-month-long series of seminars and practical instructional tools on how to improve quality assurance for future projects at this center. Data were collected on participants' designation (nurse/doctor/staff), years of work at the center, whether had prior experience with QI projects, and the score on the knowledge and attitude test taken at the end of the study. The score was constructed as the sum of 20 questions on a 5-point Likert scale, thus potentially ranging between 20 and 100. The large value indicates better knowledge about QI and more confidence and desire to use it in upcoming projects.

**Read CSV file into Colab.**
```R
# Read .cvs file into Colab in R code from GitHub
QI.data <- read_csv('https://raw.githubusercontent.com/coronasanchez/LACC-MATH-239/main/HW2/Exercise2.2Data.csv')

attach(QI.data)
head(QI.data)
```

**Plot the histogram for the scores (y) and conduct the normality tests.**
```R
QI_hist <- ggplot(QI.data, aes(x = score)) +
  geom_histogram(aes(y = after_stat(density)), bins = 8, fill ="gray",color = "black") +
    stat_function(fun = dnorm, args = list(mean = mean(score), sd = sd(score)), color = "red", linewidth = 1) +
      labs(title = "The difference in QI score with Normal Distribution Line with fitted line", x = "Diff OI Score", y = "Relative Frequency") +
        theme_bw()

# Print Histogram
print(QI_hist)

# Print Shapiro's Test
shapiro.test(QI.data$score)
```
_Cell Output:_
```
 
	Shapiro-Wilk normality test

data:  QI.data$score
W = 0.94357, p-value = 0.02913
```

<p align="center">
  <img width="400" height="auto" alt="image" src="https://github.com/user-attachments/assets/80648b34-e8ac-42fc-9679-23638b7b7a71" />
</p>

Let us tranform the data to be normal distributed utilzing Box-Cox.
```R
# applying box-cox transformation with lambda = 1-1/(y)
QI.score.tr <- 1 - 1/(re_score)

# print to validate
# QI.score.tr

QI_hist.tr <- ggplot(QI.data, aes(x = QI.score.tr)) +
  geom_histogram(aes(y = after_stat(density)), bins = 8, fill = "lightblue", color = "black") +
    stat_function(fun = dnorm, args = list(mean = mean(QI.score.tr), sd = sd(QI.score.tr)), color = "red", linewidth = 1) +
      labs(title = "QI score with Normal Distribution Line after Box-Cox Transformation", x = "Diff OI Score", y = "Relative Frequency") +
        theme_bw()

print(QI_hist.tr)

shapiro.test(QI.score.tr)
```
_Cell output:_
```
 
	Shapiro-Wilk normality test

data:  QI.score.tr
W = 0.96606, p-value = 0.2073
```
<p align="center">
  <img width="400" height="auto" alt="image" src="https://github.com/user-attachments/assets/2fdabee8-34ab-4e04-91c1-df3a4eebdacc" />
</p>  

**Fit a general linear model.**
```R
# Fit the generalized linear model to transformed response
QI.fitted.model <- glm(QI.score.tr ~ QI_desgn.relf + wrkyrs + QI_priorQI.relf, data = QI.data, family = gaussian(link = identity))

# Print the summary of the model
summary(QI.fitted.model)

# outputting estimated sigma
cat("\nsigma:\t", sigma(QI.fitted.model))
```
_Cell output:_
```
 
Call:
glm(formula = QI.score.tr ~ QI_desgn.relf + wrkyrs + QI_priorQI.relf, 
    family = gaussian(link = identity), data = QI.data)

Coefficients:
                     Estimate Std. Error t value Pr(>|t|)    
(Intercept)         -0.609290   0.094150  -6.471 1.03e-07 ***
QI_desgn.relfdoctor  0.179873   0.091507   1.966   0.0563 .  
QI_desgn.relfnurse   0.212200   0.087173   2.434   0.0195 *  
wrkyrs               0.000243   0.004086   0.059   0.9529    
QI_priorQI.relfyes   0.077263   0.068323   1.131   0.2649    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 0.04317095)

    Null deviance: 2.0290  on 44  degrees of freedom
Residual deviance: 1.7268  on 40  degrees of freedom
AIC: -7.0122

Number of Fisher Scoring iterations: 2

sigma:	 0.2077762
```
**Reduced fitted model:** </br>
$\widehat{\mathbb{E}}(1 - \frac{1}{score}) = -0.609290 + 0.179873 ⋅ (Doctor) + 0.212200 ⋅ (nurse)$

**Model Fits the data well?**
```R
# checking model fit
QI_null.model <-
  glm(QI.score.tr ~ 1, data = QI.data, family = gaussian(link = identity))

# Get total num of coeff
num_coeff_QI_df <- length(coef(QI.fitted.model)) - 1
cat("total num of DF:", num_coeff_QI_df, "\n")

QI.deviance <- -2*(logLik(QI_null.model)-logLik(QI.fitted.model))
cat("deviance =\t", QI.deviance,"\n")

QI.p.value <- pchisq(QI.deviance, df = num_coeff_QI_df, lower.tail = FALSE)
cat("p-value =\t", QI.p.value, "\n")
```
_Cell output:_
```
total num of DF: 4 
deviance =	 7.256887 
p-value =	 0.1229198 
```
Predict the score for a nurse who has worked at the center for seven years and who had previously been a co-PI on a grant that involved a quality assurance component.

$score^0 = 100 * [1-(-0.609290 + 0.179873 ⋅ (0) + 0.212200 ⋅ (1) + 0.000243 ⋅ (7) + 0.077263 ⋅ (1))]^{-1}$
$score^0 = 100 * 0.75865 = 75.865$
```R
QI_predict.tr <- predict(QI.fitted.model, data.frame(QI_desgn.relf = "nurse", wrkyrs = 7, QI_priorQI.relf = "yes"))
# Overwrite
QI_predict.tr <- 100/(1-QI_predict.tr)
cat("QI Prediction:", QI_predict.tr )
```
_Cell output:_
```
QI Prediction: 75.86528
```

## Example 3
A health insurance firm is analyzing the aggregate insurance claims that were received in a particular fiscal year. Investigators randomly select 40 companies that are insured by this firm and, for each company, pull out the data on the number of policies, the number of years insured with the firm, the percent of open claims from last year, and the aggregate claim amount from this year (in millions of dollars). 

**Read CSV file into Colab.**
```R
# Read .cvs file into Colab in R code from GitHub
HealthClaims.data <- read_csv('https://raw.githubusercontent.com/coronasanchez/LACC-MATH-239/main/HW2/Exercise2.4Data.csv')

attach(HealthClaims.data)
head(HealthClaims.data)
```

**Plot the histogram for the scores (y) and conduct the normality tests.**
```R
# resc_npolicies <- health.claims.data$npolicies / 1000
HealthClaims_hist <- ggplot(HealthClaims.data, aes(x = claimamount)) +
  geom_histogram(aes(y = after_stat(density)), bins = 7, fill ="gray",color = "black") +
    stat_function(fun = dnorm, args = list(mean = mean(claimamount), sd = sd(claimamount)), color = "red", linewidth = 1) +
      labs(title = "The difference in Num of Policies with Normal Distribution Line with fitted line", x = "Diff Num of Policies", y = "Relative Frequency") +
        theme_bw()

print(HealthClaims_hist)

shapiro.test(claimamount)
```
_Cell output:_
```
 
	Shapiro-Wilk normality test

data:  claimamount
W = 0.85595, p-value = 0.0001259
```

<p align="center">
  <img width="400" height="auto" alt="image" src="https://github.com/user-attachments/assets/3dc47955-675d-463d-abcd-d2fcc5a2f82e" />
</p>  

```R
# Finding optimal lambda for Box-Cox transformation
HealthClaims.bxcx.fit <- boxcox(claimamount ~ npolicies + yrswithfirm + percopenclaims, data = HealthClaims.data, lambda = seq(-3,3,1/4), interp = FALSE)
HealthClaims.bxcx.data <- data.frame(HealthClaims.bxcx.fit$x, HealthClaims.bxcx.fit$y)
HealthClaims.ordered.data <- HealthClaims.bxcx.data[with(HealthClaims.bxcx.data, order(-HealthClaims.bxcx.fit.y)),]

# Display
HealthClaims.ordered.data[1,]
```
<p align="center">
  <img width="400" height="auto" alt="image" src="https://github.com/user-attachments/assets/34f58501-79e6-48f2-9cea-86ff4ecbcc57" />

</p>

```R
# applying box-cox transformation with lambda = 0.5
claimamount.tr <- 2 * (sqrt(claimamount) - 1)

# plotting histogram for tranformed response
HealthClaims_hist.tr <- ggplot(HealthClaims.data, aes(x = claimamount.tr)) +
  geom_histogram(aes(y = after_stat(density)), bins = 8, fill = "lightblue", color = "black") +
    stat_function(fun = dnorm, args = list(mean = mean(claimamount.tr), sd = sd(claimamount.tr)), color = "red", linewidth = 1) +
      labs(title = "Claim Amount with Normal Distribution Line after Box-Cox Transformation", x = "Claim Amount", y = "Relative Frequency") +
        theme_bw()

print(HealthClaims_hist.tr)

shapiro.test(claimamount.tr)
```
_Cell output:_
```
 
	Shapiro-Wilk normality test

data:  claimamount.tr
W = 0.97601, p-value = 0.5445
```
<p align="center">
  <img width="400" height="auto" alt="image" src="https://github.com/user-attachments/assets/acbbeb2d-602c-430e-ac68-f7afbc5b85ff" />
</p>

**Fit a general linear model.**
```R
# Fit the generalized linear model to transformed response
HealthClaims.fitted.model <- glm(claimamount.tr ~ npolicies + yrswithfirm + percopenclaims, data = HealthClaims.data, family = gaussian(link = identity))

# Print the summary of the model
summary(HealthClaims.fitted.model)

# outputting estimated sigma
cat("\nsigma:\t\t", sigma(HealthClaims.fitted.model))
```
_Cell output:_
```
 
Call:
glm(formula = claimamount.tr ~ npolicies + yrswithfirm + percopenclaims, 
    family = gaussian(link = identity), data = HealthClaims.data)

Coefficients:
                 Estimate Std. Error t value Pr(>|t|)    
(Intercept)     0.6927512  5.5064582   0.126    0.901    
npolicies       0.0006624  0.0001413   4.688 3.89e-05 ***
yrswithfirm    -0.2337924  0.1882646  -1.242    0.222    
percopenclaims -0.0825362  0.1496114  -0.552    0.585    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 52.11152)

    Null deviance: 3064.4  on 39  degrees of freedom
Residual deviance: 1876.0  on 36  degrees of freedom
AIC: 277.44

Number of Fisher Scoring iterations: 2

sigma:		 7.218831
```
**Reduced Fitted Model:**</br>
$\widehat{\mathbb{E}}(2(sqrt(ClaimAmount)-1)) = 0.0006624 ⋅ (Num Of Policies)$

**Model Fits the data well?**
```R
# checking model fit
HealthClaims_null.model <-
  glm(claimamount.tr ~ 1, data = HealthClaims.data, family = gaussian(link = identity))

# Get total num of coeff
num_coeff_healthcare_df <- length(coef(HealthClaims.fitted.model)) - 1
cat("total num of DF:", num_coeff_healthcare_df, "\n")

HealthClaims.deviance <- -2*(logLik(HealthClaims_null.model)-logLik(HealthClaims.fitted.model))
cat("deviance =\t", HealthClaims.deviance,"\n")

HealthClaims.pvalue <- pchisq(HealthClaims.deviance, df = num_coeff_healthcare_df, lower.tail = FALSE)
cat("p-value =\t", HealthClaims.pvalue, "\n")
```
_Cell output:_
```
total num of DF: 3 
deviance =	 19.62872 
p-value =	 0.0002026295 
```

Compute the predicted amount of aggregate claims for a company with 15,500 policyholders, that has been buying policies at this firm for the past three years, and that still has 15\% of outstanding claims from the previous year.

$ClaimAmount^0 = [1/2 * (0.692751 + 0.0006624 ⋅ (15500) - 0.2337924 ⋅ (3) - 0.0825362 ⋅ (15)) + 1]^2$
$ClaimAmount^0 = [1/2*9.0205308 + 1]^2$ </br>
$ClaimAmount^0 = [4.5102654 + 1]^2 = 30.363$

```R
claimamount_predict.tr <- predict(HealthClaims.fitted.model, data.frame(npolicies = 15500, yrswithfirm = 3, percopenclaims = 15))

# Overwrite
claimamount_predict.tr <- (claimamount_predict.tr/2 + 1)**2
cat("Health Claims Prediction:", claimamount_predict.tr)
```
_Cell output:_
```
Health Claims Prediction: 30.36
```
