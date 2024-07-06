# Data:-

```
- Dataset having with 200 rows and 4 columns ['TV', 'radio', 'newspaper', 'sales'].
- In this dataset, all columns are in the same data type which is nothing but all are int types.
- In this dataset, we are not having any null values.
- The TV column having hight correlation (0.7822) with Sales column compare with other columns.
- The Radio column having (0.5762) % correlation with Sales.

```

## Tv_Column:-

```
- The TV column having with -0.069 skewness. which is we consider as a normal distribution. Its a left side skewness. if we want to apply Transformation technique weneed to apply Exponential, Power or Boxcox transformation.
- TV and Sales having 78% of correlation.
```

## Radio_column:-

```
- The Radio column having with 0.094 skewness. which is we consider as a normal distribution.Its a right side skewness. if we want to apply Transformation technique weneed to apply Log, Root or Boxcox transformation.
- Radio and Sales having 58% of Correlation.

```

## Newspaper_column:-

```
- The Newspaper column having with 0.894 skewness. which is we consider as a normal distribution. if we want to apply Transformation technique weneed to apply Log, Root or Boxcox transformation.
- The Newspapere column having some Outliers as well.
- Newspaper and Sales ahving 23% of Correlation.
```


## EDA:-

```
- In EDA process, We find some outliers in Newspaper column and replace them with lower limit and upper limit value.
- We did not find any null values in any one of the columns.
```

## Data Cleaning:-
```
- Data Cleaning process, We replace the outliers with lower limit and upper limit value.
```

## Data Wrangling:-
```
- We find some skewness in all cloumns and applyed feature transformation technique to make the data into symmetrical distribution.
- For regression model we wont apply the feature scaling technique. With applying and without applying the feature scaling technique we will get same result for regression model.
```

## Feature Selection:-
```
- In feature selection, we apply the wrapping method to find out the relationship between all 3 independent features and dependent feature based on p-values as well as we will get the R2 and adj-R2 values by applying the wrapping method.
- In wrapping method we applyed the OLS method.
- After applying the wrapping method, we abserve the relationship between independent features and dependent feature. The newspaper column is not having the relationship with dependent features which means the p-value is 0.835, nothing but we reject the null hypothesis.
- We got R2 value is 0.888 and adj-R2 value is 0.886.
==============================================================================
Dep. Variable:                      y   R-squared:                       0.888
Model:                            OLS   Adj. R-squared:                  0.887
Method:                 Least Squares   F-statistic:                     520.1
Date:                Mon, 20 May 2024   Prob (F-statistic):           4.96e-93
Time:                        15:14:04   Log-Likelihood:                -1776.0
No. Observations:                 200   AIC:                             3560.
Df Residuals:                     196   BIC:                             3573.
Df Model:                           3
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   2990.3417    404.940      7.385      0.000    2191.742    3788.941
X[0]           0.0145      0.000     31.253      0.000       0.014       0.015
X[1]           0.5515      0.026     21.093      0.000       0.500       0.603
X[2]           0.4635      2.220      0.209      0.835      -3.914       4.841
==============================================================================
Omnibus:                       57.946   Durbin-Watson:                   2.074
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              139.267
Skew:                          -1.290   Prob(JB):                     5.74e-31
Kurtosis:                       6.171   Cond. No.                     1.67e+06
==============================================================================

- By applying OLS method on 2 independent features, we got R2 value is 0.888 and adj-R2 value is 0.887.

==============================================================================
Dep. Variable:                     y1   R-squared:                       0.888
Model:                            OLS   Adj. R-squared:                  0.887
Method:                 Least Squares   F-statistic:                     783.9
Date:                Mon, 20 May 2024   Prob (F-statistic):           1.60e-94
Time:                        15:14:04   Log-Likelihood:                -1776.0
No. Observations:                 200   AIC:                             3558.
Df Residuals:                     197   BIC:                             3568.
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   3045.1445    307.602      9.900      0.000    2438.530    3651.759
X1[0]          0.0145      0.000     31.351      0.000       0.014       0.015
X1[1]          0.5532      0.025     22.288      0.000       0.504       0.602
==============================================================================
Omnibus:                       58.454   Durbin-Watson:                   2.079
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              142.245
Skew:                          -1.296   Prob(JB):                     1.29e-31
Kurtosis:                       6.218   Cond. No.                     1.27e+06
==============================================================================

```

## Feature Engineering:-
```
- In feature engineering, we are not going to apply PCA method to reduce the size of the data. which means reduce the number of features.
```

## VIF:-
```
- Applyed VIF method on 2 independent features but we did not find any multicollinearity.

        VIF    features
0  2.269759    tv_power
1  2.269759  radio_root

```

# Polynomial Regression Algorithm:-
```
- By applying Polynomial Regression algorithm on 2 independent features (which is consider as Multiple Polynomial Regression), we got train & test scores those are respectively 0.982 & 0.986.
- In Polynomial regression algorithm, we took the Degree value as 3.

Best degree train value : 3
****************************************
Best degree test value : 3

- As of now, We can consider this model as a best model with +/- 5% train and test score.

Polynomial Training Score : 0.9892136035502934
****************************************
Polynomial Test Score : 0.990922787724973

```

```

# Model Saving:-
```
- By using Pickle file, we saved the model and predicted with new data.
```