# Exploring Your Data - Lab

## Introduction 

In this lab, you'll perform an EDA task, using your skills with statistics and data visualizations. You'll continue using the Lego dataset that you've acquired and cleaned in the previous labs. 

## Objectives
You will be able to:

* Examine the descriptive statistics of our data set
* Create visualizations to better understand the distributions of variables in a dataset

## Data Exploration

At this point, you've already done a modest amount of EDA between investigating the initial dataset to further exploring individual features while cleaning things up in preparation for modeling. During this process, you've become more familiar with the particular idiosyncrasies of the dataset. This gives you an opportunity to uncover difficulties and potential pitfalls in working with the dataset as well as potential avenues for feature engineering that could improve the predictive performance of your model down the line. Remember that this is also not a linear process; after building an initial model, you might go back and continue to mine the dataset for potential inroads to create additional features and improve the model's performance if the initial results did not satisfy your needs and expectations. Here, you'll continue this process, investigating the distributions of some of the various features and their relationship to the target variable: `list_price`.

In the cells below: 

* Import `pandas` and set the standard alias. 
* Import `numpy` and set the standard alias. 
* Import `matplotlib.pyplot` and set the standard alias. 
* Import `seaborn` and set the alias `sns` (this is the standard alias for seaborn). 
* Use the ipython magic command to set all matplotlib visualizations to display inline in the notebook. 
* Load the dataset stored in the `'Lego_data_cleaned.csv'` file into a DataFrame. 
* Inspect the head of the DataFrame to ensure everything loaded correctly. 


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
# __SOLUTION__ 
import warnings
warnings.filterwarnings('ignore')
```


```python
# Import libraries and load Lego_data_merged.csv
```


```python
# __SOLUTION__ 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use('seaborn')
df = pd.read_csv('Lego_dataset_cleaned.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>piece_count</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>play_star_rating</th>
      <th>star_rating</th>
      <th>val_star_rating</th>
      <th>ages_10+</th>
      <th>ages_10-14</th>
      <th>ages_10-16</th>
      <th>ages_10-21</th>
      <th>...</th>
      <th>country_NZ</th>
      <th>country_PL</th>
      <th>country_PT</th>
      <th>country_US</th>
      <th>review_difficulty_Average</th>
      <th>review_difficulty_Challenging</th>
      <th>review_difficulty_Easy</th>
      <th>review_difficulty_Very Challenging</th>
      <th>review_difficulty_Very Easy</th>
      <th>review_difficulty_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.273020</td>
      <td>29.99</td>
      <td>-0.398512</td>
      <td>-0.655279</td>
      <td>-0.045687</td>
      <td>-0.365010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.404154</td>
      <td>19.99</td>
      <td>-0.398512</td>
      <td>-0.655279</td>
      <td>0.990651</td>
      <td>-0.365010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.517242</td>
      <td>12.99</td>
      <td>-0.147162</td>
      <td>-0.132473</td>
      <td>-0.460222</td>
      <td>-0.204063</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.635296</td>
      <td>99.99</td>
      <td>0.187972</td>
      <td>-1.352353</td>
      <td>0.161581</td>
      <td>0.117830</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.288812</td>
      <td>79.99</td>
      <td>-0.063378</td>
      <td>-2.049427</td>
      <td>0.161581</td>
      <td>-0.204063</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 103 columns</p>
</div>



- Describe the dataset using 5-point statistics. 


```python
# Your code here
```


```python
# __SOLUTION__ 
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>piece_count</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>play_star_rating</th>
      <th>star_rating</th>
      <th>val_star_rating</th>
      <th>ages_10+</th>
      <th>ages_10-14</th>
      <th>ages_10-16</th>
      <th>ages_10-21</th>
      <th>...</th>
      <th>country_NZ</th>
      <th>country_PL</th>
      <th>country_PT</th>
      <th>country_US</th>
      <th>review_difficulty_Average</th>
      <th>review_difficulty_Challenging</th>
      <th>review_difficulty_Easy</th>
      <th>review_difficulty_Very Challenging</th>
      <th>review_difficulty_Very Easy</th>
      <th>review_difficulty_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.087000e+04</td>
      <td>10870.000000</td>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>...</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.287856e-17</td>
      <td>67.309137</td>
      <td>1.340030e-17</td>
      <td>3.505388e-14</td>
      <td>2.523956e-13</td>
      <td>-1.584433e-13</td>
      <td>0.049126</td>
      <td>0.001932</td>
      <td>0.013615</td>
      <td>0.016927</td>
      <td>...</td>
      <td>0.046274</td>
      <td>0.043330</td>
      <td>0.044618</td>
      <td>0.066421</td>
      <td>0.308832</td>
      <td>0.091536</td>
      <td>0.351978</td>
      <td>0.001932</td>
      <td>0.083257</td>
      <td>0.162466</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000000e+00</td>
      <td>94.669414</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.216141</td>
      <td>0.043913</td>
      <td>0.115894</td>
      <td>0.129005</td>
      <td>...</td>
      <td>0.210088</td>
      <td>0.203609</td>
      <td>0.206474</td>
      <td>0.249029</td>
      <td>0.462033</td>
      <td>0.288384</td>
      <td>0.477609</td>
      <td>0.043913</td>
      <td>0.276282</td>
      <td>0.368894</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-6.050659e-01</td>
      <td>2.272400</td>
      <td>-4.264402e-01</td>
      <td>-5.883334e+00</td>
      <td>-5.641909e+00</td>
      <td>-5.193413e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-4.895715e-01</td>
      <td>21.899000</td>
      <td>-3.705846e-01</td>
      <td>-4.810100e-01</td>
      <td>-4.602216e-01</td>
      <td>-3.650101e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-3.379852e-01</td>
      <td>36.587800</td>
      <td>-2.868011e-01</td>
      <td>2.160641e-01</td>
      <td>1.615809e-01</td>
      <td>1.178302e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.263593e-02</td>
      <td>73.187800</td>
      <td>-1.192341e-01</td>
      <td>5.646012e-01</td>
      <td>7.833834e-01</td>
      <td>6.006705e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.466055e+00</td>
      <td>1104.870000</td>
      <td>9.795146e+00</td>
      <td>1.087407e+00</td>
      <td>9.906510e-01</td>
      <td>1.244458e+00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 103 columns</p>
</div>



- Use pandas to plot histograms for all the numeric variables in the dataset. 


```python
# Your code here
```


```python
# __SOLUTION__ 
df.hist(figsize = (20,18));
```


    
![png](index_files/index_11_0.png)
    


Note how skewed most of these distributions are. While linear regression does not assume that each of the individual predictors are normally distributed, it does assume a linear relationship between the predictors and the target variable (`list_price` in this case). To further investigate if this assumption holds true, you can plot some single variable regression plots of each feature against the target variable using `seaborn`. 

## Check for Linearity

Recall that one assumption in linear regression is that the target variable is linearly related to the input features. As shown in the previous lesson, you can use the `sns.jointplot()` function to investigate whether this relation holds true for the various predictors on hand.


```python
# Your code here
```


```python
# __SOLUTION__ 
sns.jointplot('piece_count','list_price', data=df, kind='reg');
```


    
![png](index_files/index_14_0.png)
    



```python
# __SOLUTION__
# Comment: piece_count seems to have a linear relationship with list_price
```


```python
# Your code here
```


```python
# __SOLUTION__ 
sns.jointplot('num_reviews','list_price', data=df, kind='reg');
```


    
![png](index_files/index_17_0.png)
    



```python
# __SOLUTION__
# Comment: There seems to be a some-what linear correlation between num_reviews and list_price
# Though the relationship is noisier than what we saw with piece_count
```


```python
# Your code here
```


```python
# __SOLUTION__ 
sns.jointplot('play_star_rating','list_price', data=df, kind='reg');
```


    
![png](index_files/index_20_0.png)
    



```python
# __SOLUTION__
# Comment: play_star_rating doesn't seem to have much of a linear relationship 
# with list_price
```


```python
# Your code here
```


```python
# __SOLUTION__ 
sns.jointplot('star_rating', 'list_price', data=df, kind='reg');
```


    
![png](index_files/index_23_0.png)
    



```python
# __SOLUTION__
# Comment: Again, little to no linear relation.
```


```python
# Your code here
```


```python
# __SOLUTION__ 
sns.jointplot("val_star_rating", "list_price", data=df, kind="reg");
```


    
![png](index_files/index_26_0.png)
    



```python
# __SOLUTION__
# Comment: Again, little to no linear relation.
```


```python
# __SOLUTION__
# Comments:
# Well, at first look it appears that the previous efforts in order to fill in the null review values proved of little value. 
# Perhaps this was due to imputing the mean, but as it currently stands, each of the rating features seems to have little 
# to no predictive power for the upcoming model.
```

## Checking for Multicollinearity

It's also important to make note of whether your predictive features will result in multicollinearity in the resulting model. While definitive checks for multicollinearity require analyzing the resulting model, predictors with overly high pairwise-correlation (r > .65) are almost certain to produce multicollinearity in a model. With that, take a minute to generate the pairwise (pearson) correlation coefficients of your predictive features and visualize these coefficients as a heatmap.


```python
# Your code here
```


```python
# __SOLUTION__ 
# Your code here
feats = ['piece_count', 'num_reviews', 'play_star_rating','star_rating','val_star_rating']
corr = df[feats].corr()
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>piece_count</th>
      <th>num_reviews</th>
      <th>play_star_rating</th>
      <th>star_rating</th>
      <th>val_star_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>piece_count</th>
      <td>1.000000</td>
      <td>0.548783</td>
      <td>-0.023281</td>
      <td>0.055481</td>
      <td>0.057313</td>
    </tr>
    <tr>
      <th>num_reviews</th>
      <td>0.548783</td>
      <td>1.000000</td>
      <td>-0.070892</td>
      <td>-0.002466</td>
      <td>0.020471</td>
    </tr>
    <tr>
      <th>play_star_rating</th>
      <td>-0.023281</td>
      <td>-0.070892</td>
      <td>1.000000</td>
      <td>0.619044</td>
      <td>0.485843</td>
    </tr>
    <tr>
      <th>star_rating</th>
      <td>0.055481</td>
      <td>-0.002466</td>
      <td>0.619044</td>
      <td>1.000000</td>
      <td>0.728203</td>
    </tr>
    <tr>
      <th>val_star_rating</th>
      <td>0.057313</td>
      <td>0.020471</td>
      <td>0.485843</td>
      <td>0.728203</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Your code here
```


```python
# __SOLUTION__ 
sns.heatmap(corr, center=0, annot=True);
```


    
![png](index_files/index_33_0.png)
    



```python
# __SOLUTION__
# Comments: 
# The rating features show little promise for adding predictive power towards the `list_price`. 
# This diminishes worry concerning their high correlation. 
# That said, the two most promising predictors: `piece_count` and `num_reviews` also display fairly high correlation. 
# Further analysis of an initial model will clearly be warranted.
```

## Further Resources

Have a look at following resources on how to deal with complex datasets that don't meet our initial expectations:  

- [What to Do When Bad Data Thwarts Machine Learning Success](https://towardsdatascience.com/what-to-do-when-bad-data-thwarts-machine-learning-success-fb82249aae8b)

- [Practical advice for analysis of large, complex data sets ](http://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html)

- [Data Cleaning Challenge: Scale and Normalize Data](https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data)

## Summary 

In this lesson you performed some initial EDA using descriptive statistics and data visualizations to check for regression assumptions. In the upcoming lessons, you'll continue to carry out a standard Data Science process and begin to fit and refine an initial model.
