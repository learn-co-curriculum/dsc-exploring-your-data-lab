
# Exploring Your Data - Lab

## Introduction 

In this lab you'll perform a exploratory data analysis task, using statistical and visual EDA skills. You'll continue using the Lego dataset that you've acquired and cleaned in the previous labs. 

## Objectives
You will be able to:

* Check the distribution of various columns
* Examine the descriptive statistics of our data set
* Create visualizations to help us better understand our data set

## Data Exploration

At this point, you've already done a modest amount of data exploration between investigating the initial database to further exploring individual features while cleaning things up in preparation for modeling. During this process, you've become more familiar with the particular idiosyncrasies of the dataset. This gives you an opportunity to uncover difficulties and potential pitfalls in working with the dataset as well as potential avenues for feature engineering that could improve the predictive performance of your model down the line. Remember that this is also not a linear process; after building an initial model, you might go back and continue to mine the dataset for potential inroads to create additional features and improve the model's performance if initial results did not satisfy your needs and expectations. Here, you'll continue this process, investigating the distributions of some of the various features and their relationship to the target variable: `list_price`.

### Load the dataset 'Lego_dataset_cleaned.csv'  and Check its Contents 


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
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
      <th>prod_id</th>
      <th>ages</th>
      <th>piece_count</th>
      <th>set_name</th>
      <th>prod_desc</th>
      <th>prod_long_desc</th>
      <th>theme_name</th>
      <th>country</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>play_star_rating</th>
      <th>review_difficulty</th>
      <th>star_rating</th>
      <th>val_star_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75823</td>
      <td>6-12</td>
      <td>-0.273020</td>
      <td>Bird Island Egg Heist</td>
      <td>Catapult into action and take back the eggs fr...</td>
      <td>Use the staircase catapult to launch Red into ...</td>
      <td>Angry Birds™</td>
      <td>US</td>
      <td>29.99</td>
      <td>-0.398512</td>
      <td>-0.655279</td>
      <td>Average</td>
      <td>-0.045687</td>
      <td>-0.365010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>75822</td>
      <td>6-12</td>
      <td>-0.404154</td>
      <td>Piggy Plane Attack</td>
      <td>Launch a flying attack and rescue the eggs fro...</td>
      <td>Pilot Pig has taken off from Bird Island with ...</td>
      <td>Angry Birds™</td>
      <td>US</td>
      <td>19.99</td>
      <td>-0.398512</td>
      <td>-0.655279</td>
      <td>Easy</td>
      <td>0.990651</td>
      <td>-0.365010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75821</td>
      <td>6-12</td>
      <td>-0.517242</td>
      <td>Piggy Car Escape</td>
      <td>Chase the piggy with lightning-fast Chuck and ...</td>
      <td>Pitch speedy bird Chuck against the Piggy Car....</td>
      <td>Angry Birds™</td>
      <td>US</td>
      <td>12.99</td>
      <td>-0.147162</td>
      <td>-0.132473</td>
      <td>Easy</td>
      <td>-0.460222</td>
      <td>-0.204063</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21030</td>
      <td>12+</td>
      <td>0.635296</td>
      <td>United States Capitol Building</td>
      <td>Explore the architecture of the United States ...</td>
      <td>Discover the architectural secrets of the icon...</td>
      <td>Architecture</td>
      <td>US</td>
      <td>99.99</td>
      <td>0.187972</td>
      <td>-1.352353</td>
      <td>Average</td>
      <td>0.161581</td>
      <td>0.117830</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21035</td>
      <td>12+</td>
      <td>0.288812</td>
      <td>Solomon R. Guggenheim Museum®</td>
      <td>Recreate the Solomon R. Guggenheim Museum® wit...</td>
      <td>Discover the architectural secrets of Frank Ll...</td>
      <td>Architecture</td>
      <td>US</td>
      <td>79.99</td>
      <td>-0.063378</td>
      <td>-2.049427</td>
      <td>Challenging</td>
      <td>0.161581</td>
      <td>-0.204063</td>
    </tr>
  </tbody>
</table>
</div>



### Describe the dataset using 5 point statistics and record your observations


```python
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
      <th>prod_id</th>
      <th>piece_count</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>play_star_rating</th>
      <th>star_rating</th>
      <th>val_star_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
      <td>10870.000000</td>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.181634e+04</td>
      <td>1.154959e-16</td>
      <td>67.309137</td>
      <td>3.087316e-16</td>
      <td>3.548158e-14</td>
      <td>2.524533e-13</td>
      <td>-1.584896e-13</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.736390e+05</td>
      <td>1.000000e+00</td>
      <td>94.669414</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.300000e+02</td>
      <td>-6.050659e-01</td>
      <td>2.272400</td>
      <td>-4.264402e-01</td>
      <td>-5.883334e+00</td>
      <td>-5.641909e+00</td>
      <td>-5.193413e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.112300e+04</td>
      <td>-4.895715e-01</td>
      <td>21.899000</td>
      <td>-3.705846e-01</td>
      <td>-4.810100e-01</td>
      <td>-4.602216e-01</td>
      <td>-3.650101e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.207350e+04</td>
      <td>-3.379852e-01</td>
      <td>36.587800</td>
      <td>-2.868011e-01</td>
      <td>2.160641e-01</td>
      <td>1.615809e-01</td>
      <td>1.178302e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.124800e+04</td>
      <td>6.263593e-02</td>
      <td>73.187800</td>
      <td>-1.192341e-01</td>
      <td>5.646012e-01</td>
      <td>7.833834e-01</td>
      <td>6.006705e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000431e+06</td>
      <td>8.466055e+00</td>
      <td>1104.870000</td>
      <td>9.795146e+00</td>
      <td>1.087407e+00</td>
      <td>9.906510e-01</td>
      <td>1.244458e+00</td>
    </tr>
  </tbody>
</table>
</div>



### Use pandas histogram plotting to plot histograms for all the variables in the dataset


```python
df.hist(figsize = (20,18));
```


![png](index_files/index_6_0.png)


Note how skewed most of these distributions are. While linear regression does not assume that each of the individual predictors are normally distributed, it does assume a linear relationship between the predictors and the target variable (list_price in this case). To further investigate if this assumption holds true, you can plot some single variable regression plots of each feature against the target variable using seaborn.

## Check for Linearity

Recall that one assumption in linear regression is that the target variable is linearly related to the input features. As shown in the previous lesson, you can use the `sns.jointplot()` function to investigate whether this relation holds true for the various predictors on hand.


```python
sns.jointplot("piece_count","list_price", data=df, kind="reg");
```


![png](index_files/index_8_0.png)



```python
sns.jointplot("num_reviews","list_price", data=df, kind="reg");
```


![png](index_files/index_9_0.png)



```python
sns.jointplot("play_star_rating","list_price", data=df, kind="reg");
```


![png](index_files/index_10_0.png)


> *Comment:* Play start rating doesn't seem to have much of a linear relationship with list_price  


```python
sns.jointplot("star_rating", "list_price", data=df, kind="reg");
```


![png](index_files/index_12_0.png)


> *Comment:* Again little to no linear relation.


```python
sns.jointplot("val_star_rating", "list_price", data=df, kind="reg");
```


![png](index_files/index_14_0.png)


> *Comment:* Again little to no linear relation.

## Comments

Well, at first look it appears that the previous efforts in order to fill in the null review values proved of little value. Perhaps this was due to imputing the mean, but as it currently stands, each of the rating features seems to have little to no predictive power for the upcoming model.

## Checking for Multicollinearity

It's also important to make note of whether your predictive features will result in multicollinearity in the resulting model. While definitive checks for multicollinearity require analyzing the resulting model, predictors with overly high pairwise-correlation (r^2 > .65) are almost certain to produce multicollinearity in a model. With that, take a minute to generate the pairwise [pearson] correlation coefficients of your predictive features and visualizes these coefficients as a heatmap.


```python
#Your code here
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
sns.heatmap(corr, center=0, annot=True);
```


![png](index_files/index_18_0.png)


> Comments: The rating features show little promise for adding predictive power towards the list_price. This diminishes worry concerning their high correlation. That said, the two most promising predictors: piece_count and num_reviews also display fairly high correlation. Further analysis of an initial model will clearly be warranted.

## Further Resources

Have a look at following resources on how to deal with complex datasets that don't meet our initial expectations. 

[What to Do When Bad Data Thwarts Machine Learning Success](https://towardsdatascience.com/what-to-do-when-bad-data-thwarts-machine-learning-success-fb82249aae8b)

[Practical advice for analysis of large, complex data sets ](http://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html)

[Data Cleaning Challenge: Scale and Normalize Data](https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data)

## Summary 

In this lesson you performed some initial EDA onto check for regression assumptions. In the upcoming lessons, you'll continue to carry out a standard data science process and begin to fit and refine an initial model.
