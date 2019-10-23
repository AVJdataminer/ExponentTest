<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Data-Loading,-Data-Wrangling,-and-Quality-Checks" data-toc-modified-id="Data-Loading,-Data-Wrangling,-and-Quality-Checks-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data Loading, Data Wrangling, and Quality Checks</a></span><ul class="toc-item"><li><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Which-Vehicle-should-we-add-to-our-fleet?" data-toc-modified-id="Which-Vehicle-should-we-add-to-our-fleet?-1.0.0.1"><span class="toc-item-num">1.0.0.1&nbsp;&nbsp;</span>Which Vehicle should we add to our fleet?</a></span></li></ul></li></ul></li><li><span><a href="#Data-Loading" data-toc-modified-id="Data-Loading-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span><strong>Data Loading</strong></a></span></li><li><span><a href="#Data-Wrangling" data-toc-modified-id="Data-Wrangling-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Data Wrangling</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Missing-and-NA-Values:-Data-Science-Knowledge-Question" data-toc-modified-id="Missing-and-NA-Values:-Data-Science-Knowledge-Question-1.2.0.1"><span class="toc-item-num">1.2.0.1&nbsp;&nbsp;</span>Missing and NA Values: Data Science Knowledge Question</a></span></li></ul></li><li><span><a href="#Hint-1" data-toc-modified-id="Hint-1-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Hint 1</a></span></li><li><span><a href="#Hint-2" data-toc-modified-id="Hint-2-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>Hint 2</a></span></li><li><span><a href="#Test-your-Answer" data-toc-modified-id="Test-your-Answer-1.2.3"><span class="toc-item-num">1.2.3&nbsp;&nbsp;</span>Test your Answer</a></span></li><li><span><a href="#Missing-and-NA-Values:-Programming-Question" data-toc-modified-id="Missing-and-NA-Values:-Programming-Question-1.2.4"><span class="toc-item-num">1.2.4&nbsp;&nbsp;</span>Missing and NA Values: Programming Question</a></span></li><li><span><a href="#Our-solution" data-toc-modified-id="Our-solution-1.2.5"><span class="toc-item-num">1.2.5&nbsp;&nbsp;</span>Our solution</a></span><ul class="toc-item"><li><span><a href="#Approach" data-toc-modified-id="Approach-1.2.5.1"><span class="toc-item-num">1.2.5.1&nbsp;&nbsp;</span><strong>Approach</strong></a></span></li><li><span><a href="#Programming" data-toc-modified-id="Programming-1.2.5.2"><span class="toc-item-num">1.2.5.2&nbsp;&nbsp;</span>Programming</a></span></li></ul></li><li><span><a href="#Data-Types:-Data-Science-Knowledge-Question" data-toc-modified-id="Data-Types:-Data-Science-Knowledge-Question-1.2.6"><span class="toc-item-num">1.2.6&nbsp;&nbsp;</span>Data Types: Data Science Knowledge Question</a></span></li></ul></li></ul></li></ul></div>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sshS662wG4kdLLs3GRSQ2hrTCu_bGfnI)

# Data Loading, Data Wrangling, and Quality Checks

### How would you pre-process this data set to remove missing values?

You are a data scientist at a used vehicle dealer and your manager wants to know which vehicles are most likely to have a higher four-year resale value. You have access to a vehicle data set with many attributes associated with valuation. We can build a predictive model to determine which vehicles have higher 4-year resale values. As is often the case we need to perform pre-processing steps to handle the data before we are ready to build our predictive model. 

## **Data Loading**

**Load the python modules and mount the data drive.**


```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
import datetime
from google.colab import drive
#drive.mount('/content/drive')
```


```
#load the data
file='https://drive.google.com/uc?export=download&id=1PigHCsGrqy8IP3NthEfOa95ly1Gbwp7D'
df=pd.read_csv(file)
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Manufacturer</th>
      <th>Model</th>
      <th>Sales in thousands</th>
      <th>4-year resale value</th>
      <th>Vehicle type</th>
      <th>Price in thousands</th>
      <th>Engine size</th>
      <th>Horsepower</th>
      <th>Wheelbase</th>
      <th>Width</th>
      <th>Length</th>
      <th>Curb weight</th>
      <th>Fuel capacity</th>
      <th>Fuel efficiency</th>
      <th>Latest Launch</th>
      <th>Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acura</td>
      <td>Integra</td>
      <td>16.92</td>
      <td>16.36</td>
      <td>Passenger</td>
      <td>21.50</td>
      <td>1.8</td>
      <td>140</td>
      <td>101.2</td>
      <td>67.3</td>
      <td>172.4</td>
      <td>2.64</td>
      <td>13.2</td>
      <td>28.0</td>
      <td>2-Feb-14</td>
      <td>Japanese</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Acura</td>
      <td>TL</td>
      <td>39.38</td>
      <td>19.88</td>
      <td>Passenger</td>
      <td>28.40</td>
      <td>3.2</td>
      <td>225</td>
      <td>108.1</td>
      <td>70.3</td>
      <td>192.9</td>
      <td>3.52</td>
      <td>17.2</td>
      <td>25.0</td>
      <td>6-Mar-15</td>
      <td>Japanese</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Acura</td>
      <td>CL</td>
      <td>14.11</td>
      <td>18.23</td>
      <td>Passenger</td>
      <td>NaN</td>
      <td>3.2</td>
      <td>225</td>
      <td>106.9</td>
      <td>70.6</td>
      <td>192.0</td>
      <td>3.47</td>
      <td>17.2</td>
      <td>26.0</td>
      <td>1-Apr-14</td>
      <td>Japanese</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Acura</td>
      <td>RL</td>
      <td>8.59</td>
      <td>29.73</td>
      <td>Passenger</td>
      <td>42.00</td>
      <td>3.5</td>
      <td>210</td>
      <td>114.6</td>
      <td>71.4</td>
      <td>196.6</td>
      <td>3.85</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>3-Oct-15</td>
      <td>Japanese</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Audi</td>
      <td>A4</td>
      <td>20.40</td>
      <td>22.26</td>
      <td>Passenger</td>
      <td>23.99</td>
      <td>1.8</td>
      <td>150</td>
      <td>102.6</td>
      <td>68.2</td>
      <td>178.0</td>
      <td>3.00</td>
      <td>16.4</td>
      <td>27.0</td>
      <td>10-Aug-15</td>
      <td>European</td>
    </tr>
  </tbody>
</table>
</div>



## Data Wrangling

### Missing and NA Values: Data Science Knowledge Question
How should we handle the NA values in this cars dataset?  

Please include how we decide what to do with the missing data as well as what additional information we may want to review or consider in the process. 

Use the below code snippet and its results to help you. * Note: it may not be the correct solution.


```
print(df.shape)
print(pd.DataFrame(df.isnull().sum().sort_values(ascending=False)))
df = df.dropna(axis=1)
print(df.shape)
print(df.columns)
```

    (156, 16)
                          0
    4-year resale value  36
    Fuel efficiency       2
    Curb weight           1
    Price in thousands    1
    Region                0
    Latest Launch         0
    Fuel capacity         0
    Length                0
    Width                 0
    Wheelbase             0
    Horsepower            0
    Engine size           0
    Vehicle type          0
    Sales in thousands    0
    Model                 0
    Manufacturer          0
    (156, 12)
    Index(['Manufacturer', 'Model', 'Sales in thousands', 'Vehicle type',
           'Engine size', 'Horsepower', 'Wheelbase', 'Width', 'Length',
           'Fuel capacity', 'Latest Launch', 'Region'],
          dtype='object')


### Hint 1

When the count of NA's is less than 10% of the overall dataset it is often resonable to drop those observations. Look at the columns and counts of rows and columns and think about the percentage of missing values in the cars dataset. Is it less than 10% of the observations by column?

### Hint 2


Let's first identify our response variable; in this case it is the '4-year resale value' and therefore it cannot be removed in the cleaned dataset. Consider the printed result and which columns you would expect to be present still. 

### Test your Answer

* What are common considerations in determining how to handle missing values?
* Is there anything in particular in the approach shown that would cause problems if we next tried to build our predictive model?
* What is the size of the resulting dataframe?

### Missing and NA Values: Programming Question

Change the below code to drop only the observations with missing values and not the entire variable with missing values?


```
file='https://drive.google.com/uc?export=download&id=1PigHCsGrqy8IP3NthEfOa95ly1Gbwp7D'
df=pd.read_csv(file)
print(df.shape)
print(pd.DataFrame(df.isnull().sum().sort_values(ascending=False)))
df = df.dropna(axis=1)
print(df.shape)
print(df.columns)
```

### Our solution 

#### **Approach** 

*Identifying Missing data:* One may also want to run additional analyses to ensure there are no mask values such as '-9999' or values such as 'none', these missing values will be missed by `is.null()`. 

*Data Context:* We identify our response variable as the '4-year resale value', this column therefore must be in the cleaned output dataframe and cannot be deleted to handle the missing values in this column.   

*Missing Volume:* We identify that only 4 out 16 columns are missing data and 39 out of 156 observations or rows are missing. Three of the four columns missing data the percentages are less than 10% which is small enough to drop without having a major impact on the overall information content of the dataframe. However, our response variable is 23% missing which one could argue requires a different amelioration strategy than dropping, in addition to the fact that it is our response variable.  

*Strategy:* For the three of four columns with less than 10% missing we can simply run `df.dropna()` or `df.dropna(axis = 0)` to remove the rows with those observations only, maintaining the rest of that particular column. Notice in the example code snippet provided `df.dropna(axis = 1)` is used which drops the entire column and removes our response variable from the dataframe along with three other features that will be important in our modeling process. Printing out the dataframe before and after the dropna function helps you see what changes were made so you can be aware of spurious steps.

For the response variable, we need to return to management and ask if it is appropriate to remove those rows, or if this will result in a predictive model that is not useful because it has fewer vehicle types that can be predicted. In order to fill the missing values we impute those values. One strategy would be to group the data by 'Manufacturer' or 'region' and fill with the mean values from those groupings. This approach is decent but is only using grouping feature to inform the missing value imputation. Instead we use an imputation model to that is fit on all the numeric data in the dataframe to fill the missing values and we can apply this method to all the numeric columns in one programming step without having to drop any observations. 

If we wanted to incorporate the categorical features into our imputation model we need to convert them to indicator columns aka dummy variables first, then combined with the numeric columns we can run an impute model. Remeber, this still doesn't account for any datetime features, in our code below you will see the 'Latest Launch' is dropped because it is a datetime feature.


#### Programming 
Updated code with correct axis select in the drop statement.


```
file='https://drive.google.com/uc?export=download&id=1PigHCsGrqy8IP3NthEfOa95ly1Gbwp7D'
df=pd.read_csv(file)
print(df.shape)
print(pd.DataFrame(df.isnull().sum().sort_values(ascending=False)))
df = df.dropna(axis=0)
print(df.shape)
print(df.columns)
```

    (156, 16)
                          0
    4-year resale value  36
    Fuel efficiency       2
    Curb weight           1
    Price in thousands    1
    Region                0
    Latest Launch         0
    Fuel capacity         0
    Length                0
    Width                 0
    Wheelbase             0
    Horsepower            0
    Engine size           0
    Vehicle type          0
    Sales in thousands    0
    Model                 0
    Manufacturer          0
    (117, 16)
    Index(['Manufacturer', 'Model', 'Sales in thousands', '4-year resale value',
           'Vehicle type', 'Price in thousands', 'Engine size', 'Horsepower',
           'Wheelbase', 'Width', 'Length', 'Curb weight', 'Fuel capacity',
           'Fuel efficiency', 'Latest Launch', 'Region'],
          dtype='object')


Print the percentage of missing values for the data frame.


```
file='https://drive.google.com/uc?export=download&id=1PigHCsGrqy8IP3NthEfOa95ly1Gbwp7D'
df=pd.read_csv(file)
pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df)).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4-year resale value</th>
      <td>0.230769</td>
    </tr>
    <tr>
      <th>Fuel efficiency</th>
      <td>0.012821</td>
    </tr>
    <tr>
      <th>Curb weight</th>
      <td>0.006410</td>
    </tr>
    <tr>
      <th>Price in thousands</th>
      <td>0.006410</td>
    </tr>
    <tr>
      <th>Region</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Impute NA's without categorical data columns. Compare the results with the histograms.


```
from sklearn.preprocessing import Imputer
file='https://drive.google.com/uc?export=download&id=1PigHCsGrqy8IP3NthEfOa95ly1Gbwp7D'
df=pd.read_csv(file)
X = df.select_dtypes(exclude=['object'])
print(X.shape)
imputer = imputer.fit(X)
dfn = pd.DataFrame(imputer.transform(X), columns = X.columns)
NA_cols = ['4-year resale value','Fuel efficiency','Curb weight','Price in thousands']
```

    (156, 11)


Before Imputation


```
hist = df[NA_cols].hist(bins=50,figsize =(5,5))
```


![png](output_27_0.png)


After the numeric column fit Imputation


```
hist = dfn[NA_cols].hist(bins=50,figsize =(5, 5))
```


![png](output_29_0.png)


Impute NA's with categorical data columns. Compare the result to the original and the previously imputed dataframe.


```
file='https://drive.google.com/uc?export=download&id=1PigHCsGrqy8IP3NthEfOa95ly1Gbwp7D'
df=pd.read_csv(file)
df = df.drop(['Latest Launch'], axis =1)
dfo=df.select_dtypes(include=['object'])
fulldf = pd.concat([df.drop(dfo, axis=1), pd.get_dummies(dfo)], axis=1)
print(fulldf.shape)
imputer = imputer.fit(fulldf)
fulldfn = pd.DataFrame(imputer.transform(fulldf), columns = fulldf.columns)
```

    (156, 202)


After All Features Fit Imputation


```
hist = fulldfn[NA_cols].hist(bins=50,figsize =(5,5))
```


![png](output_33_0.png)




---



### Data Types: Data Science Knowledge Question

Assuming the NA values have been handled appropriately. We now consider the different data types in our data frame.


1.   What types of data are in this data frame? 
2.   Can we use all data types as they are in our model building step?




```
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 156 entries, 0 to 155
    Data columns (total 12 columns):
    Manufacturer          156 non-null object
    Model                 156 non-null object
    Sales in thousands    156 non-null float64
    Vehicle type          156 non-null object
    Engine size           156 non-null float64
    Horsepower            156 non-null int64
    Wheelbase             156 non-null float64
    Width                 156 non-null float64
    Length                156 non-null float64
    Fuel capacity         156 non-null float64
    Latest Launch         156 non-null object
    Region                156 non-null object
    dtypes: float64(6), int64(1), object(5)
    memory usage: 14.7+ KB



```
df.columns
```




    Index(['Manufacturer', 'Model', 'Sales in thousands', 'Vehicle type',
           'Engine size', 'Horsepower', 'Wheelbase', 'Width', 'Length',
           'Fuel capacity', 'Latest Launch', 'Region'],
          dtype='object')




```

```
