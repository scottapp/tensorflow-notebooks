# Collaborative Filtering Recommender System

In this notebook I am going to use the ecommerce dataset from Kaggle to build a user based collaborative filtering recommender system. 

The main idea is to find other similar users based on user feature vectors using cosine similarity measures. 

You can download the data at https://www.kaggle.com/carrie1/ecommerce-data, extract the zip file and save data.csv in the data directory.

### Load libraries


```python
import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile
from sklearn.metrics.pairwise import cosine_similarity
```

### Download and extract dataset


```python
#zf = ZipFile("./data/kaggle_ecommerce_data.zip") 
#zf.extractall(path = './data/') 
#zf.close()

df = pd.read_csv("./data/data.csv", encoding = 'ISO-8859-1')
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>12/1/2010 8:26</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>541904</th>
      <td>581587</td>
      <td>22613</td>
      <td>PACK OF 20 SPACEBOY NAPKINS</td>
      <td>12</td>
      <td>12/9/2011 12:50</td>
      <td>0.85</td>
      <td>12680.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>541905</th>
      <td>581587</td>
      <td>22899</td>
      <td>CHILDREN'S APRON DOLLY GIRL</td>
      <td>6</td>
      <td>12/9/2011 12:50</td>
      <td>2.10</td>
      <td>12680.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>541906</th>
      <td>581587</td>
      <td>23254</td>
      <td>CHILDRENS CUTLERY DOLLY GIRL</td>
      <td>4</td>
      <td>12/9/2011 12:50</td>
      <td>4.15</td>
      <td>12680.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>541907</th>
      <td>581587</td>
      <td>23255</td>
      <td>CHILDRENS CUTLERY CIRCUS PARADE</td>
      <td>4</td>
      <td>12/9/2011 12:50</td>
      <td>4.15</td>
      <td>12680.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>541908</th>
      <td>581587</td>
      <td>22138</td>
      <td>BAKING SET 9 PIECE RETROSPOT</td>
      <td>3</td>
      <td>12/9/2011 12:50</td>
      <td>4.95</td>
      <td>12680.0</td>
      <td>France</td>
    </tr>
  </tbody>
</table>
<p>541909 rows × 8 columns</p>
</div>



### Check data shape and columns


```python
print("Rows     : ", df.shape[0])
print("Columns  : ", df.shape[1])
print("")
print("Features : \n", df.columns.tolist())
print("")
print("Missing values :  ", df.isnull().sum().values.sum())
print("")
print("Unique values :  \n", df.nunique())
```

    Rows     :  541909
    Columns  :  8
    
    Features : 
     ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
    
    Missing values :   136534
    
    Unique values :  
     InvoiceNo      25900
    StockCode       4070
    Description     4223
    Quantity         722
    InvoiceDate    23260
    UnitPrice       1630
    CustomerID      4372
    Country           38
    dtype: int64



```python
df.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>541909.000000</td>
      <td>541909.000000</td>
      <td>406829.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.552250</td>
      <td>4.611114</td>
      <td>15287.690570</td>
    </tr>
    <tr>
      <th>std</th>
      <td>218.081158</td>
      <td>96.759853</td>
      <td>1713.600303</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-80995.000000</td>
      <td>-11062.060000</td>
      <td>12346.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.250000</td>
      <td>13953.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.080000</td>
      <td>15152.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10.000000</td>
      <td>4.130000</td>
      <td>16791.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80995.000000</td>
      <td>38970.000000</td>
      <td>18287.000000</td>
    </tr>
  </tbody>
</table>
</div>



Drop negative values and rows with invalid customer id


```python
df = df.loc[df['Quantity'] > 0]
df = df.loc[df['UnitPrice'] > 0]
```


```python
df.loc[df['CustomerID'].isna()].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1443</th>
      <td>536544</td>
      <td>21773</td>
      <td>DECORATIVE ROSE BATHROOM BOTTLE</td>
      <td>1</td>
      <td>12/1/2010 14:32</td>
      <td>2.51</td>
      <td>NaN</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1444</th>
      <td>536544</td>
      <td>21774</td>
      <td>DECORATIVE CATS BATHROOM BOTTLE</td>
      <td>2</td>
      <td>12/1/2010 14:32</td>
      <td>2.51</td>
      <td>NaN</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>536544</td>
      <td>21786</td>
      <td>POLKADOT RAIN HAT</td>
      <td>4</td>
      <td>12/1/2010 14:32</td>
      <td>0.85</td>
      <td>NaN</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>536544</td>
      <td>21787</td>
      <td>RAIN PONCHO RETROSPOT</td>
      <td>2</td>
      <td>12/1/2010 14:32</td>
      <td>1.66</td>
      <td>NaN</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>536544</td>
      <td>21790</td>
      <td>VINTAGE SNAP CARDS</td>
      <td>9</td>
      <td>12/1/2010 14:32</td>
      <td>1.66</td>
      <td>NaN</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (530104, 8)




```python
df = df.dropna(subset=['CustomerID'])
```


```python
df.shape
```




    (397884, 8)



Data should be clean now


```python
df.isnull().sum()
```




    InvoiceNo      0
    StockCode      0
    Description    0
    Quantity       0
    InvoiceDate    0
    UnitPrice      0
    CustomerID     0
    Country        0
    dtype: int64



## Create user item matrix

Our goal is to create a user item matrix where the the values in each row tell us if that particular CustomerID had purchased the item before. 


```python
user_item_matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum')
user_item_matrix.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>StockCode</th>
      <th>10002</th>
      <th>10080</th>
      <th>10120</th>
      <th>10123C</th>
      <th>10124A</th>
      <th>10124G</th>
      <th>10125</th>
      <th>10133</th>
      <th>10135</th>
      <th>11001</th>
      <th>...</th>
      <th>90214V</th>
      <th>90214W</th>
      <th>90214Y</th>
      <th>90214Z</th>
      <th>BANK CHARGES</th>
      <th>C2</th>
      <th>DOT</th>
      <th>M</th>
      <th>PADS</th>
      <th>POST</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12347.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12348.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>12349.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12350.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12352.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>12353.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12354.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12355.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12356.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 3665 columns</p>
</div>




```python
user_item_matrix = user_item_matrix.applymap(lambda x: 1 if x > 0 else 0)
user_item_matrix.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>StockCode</th>
      <th>10002</th>
      <th>10080</th>
      <th>10120</th>
      <th>10123C</th>
      <th>10124A</th>
      <th>10124G</th>
      <th>10125</th>
      <th>10133</th>
      <th>10135</th>
      <th>11001</th>
      <th>...</th>
      <th>90214V</th>
      <th>90214W</th>
      <th>90214Y</th>
      <th>90214Z</th>
      <th>BANK CHARGES</th>
      <th>C2</th>
      <th>DOT</th>
      <th>M</th>
      <th>PADS</th>
      <th>POST</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12347.0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12348.0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12349.0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12350.0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3665 columns</p>
</div>



The rows are vectors that contain items that the customer has previously bought 


```python
user_item_matrix.shape
```




    (4338, 3665)



Then we calculate the cosine similarity matrix between the rows of vectors.  Since each row is a vector that represent a particular user, we can say that the cosine similarity between the vectors may also be the similarity between each user-user pair.


```python
user_user_matrix = pd.DataFrame(cosine_similarity(user_item_matrix))
user_user_matrix
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>4328</th>
      <th>4329</th>
      <th>4330</th>
      <th>4331</th>
      <th>4332</th>
      <th>4333</th>
      <th>4334</th>
      <th>4335</th>
      <th>4336</th>
      <th>4337</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
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
      <th>1</th>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.063022</td>
      <td>0.046130</td>
      <td>0.047795</td>
      <td>0.038484</td>
      <td>0.0</td>
      <td>0.025876</td>
      <td>0.136641</td>
      <td>0.094742</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.029709</td>
      <td>0.052668</td>
      <td>0.000000</td>
      <td>0.032844</td>
      <td>0.062318</td>
      <td>0.000000</td>
      <td>0.113776</td>
      <td>0.109364</td>
      <td>0.012828</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.063022</td>
      <td>1.000000</td>
      <td>0.024953</td>
      <td>0.051709</td>
      <td>0.027756</td>
      <td>0.0</td>
      <td>0.027995</td>
      <td>0.118262</td>
      <td>0.146427</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.064282</td>
      <td>0.113961</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.170905</td>
      <td>0.083269</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.046130</td>
      <td>0.024953</td>
      <td>1.000000</td>
      <td>0.056773</td>
      <td>0.137137</td>
      <td>0.0</td>
      <td>0.030737</td>
      <td>0.032461</td>
      <td>0.144692</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.105868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039014</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.067574</td>
      <td>0.137124</td>
      <td>0.030475</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.047795</td>
      <td>0.051709</td>
      <td>0.056773</td>
      <td>1.000000</td>
      <td>0.031575</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033315</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.044866</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4333</th>
      <td>0.0</td>
      <td>0.062318</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.041523</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.105409</td>
      <td>1.000000</td>
      <td>0.119523</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4334</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.049629</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.119523</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.046613</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4335</th>
      <td>0.0</td>
      <td>0.113776</td>
      <td>0.000000</td>
      <td>0.067574</td>
      <td>0.000000</td>
      <td>0.037582</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.160128</td>
      <td>0.079305</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.174078</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.017800</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4336</th>
      <td>0.0</td>
      <td>0.109364</td>
      <td>0.170905</td>
      <td>0.137124</td>
      <td>0.044866</td>
      <td>0.080278</td>
      <td>0.0</td>
      <td>0.113354</td>
      <td>0.034204</td>
      <td>0.093170</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.037184</td>
      <td>0.016480</td>
      <td>0.043602</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.046613</td>
      <td>0.017800</td>
      <td>1.000000</td>
      <td>0.096334</td>
    </tr>
    <tr>
      <th>4337</th>
      <td>0.0</td>
      <td>0.012828</td>
      <td>0.083269</td>
      <td>0.030475</td>
      <td>0.000000</td>
      <td>0.033898</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.108324</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.104383</td>
      <td>0.000000</td>
      <td>0.043396</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.096334</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>4338 rows × 4338 columns</p>
</div>



The shape of the matrix should be square now.  The index and columns should be the corresponding customer ID


```python
user_user_matrix.shape
```




    (4338, 4338)




```python
user_user_matrix.columns = user_item_matrix.index

user_user_matrix['CustomerID'] = user_item_matrix.index

user_user_matrix = user_user_matrix.set_index('CustomerID')
user_user_matrix.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>CustomerID</th>
      <th>12346.0</th>
      <th>12347.0</th>
      <th>12348.0</th>
      <th>12349.0</th>
      <th>12350.0</th>
      <th>12352.0</th>
      <th>12353.0</th>
      <th>12354.0</th>
      <th>12355.0</th>
      <th>12356.0</th>
      <th>...</th>
      <th>18273.0</th>
      <th>18274.0</th>
      <th>18276.0</th>
      <th>18277.0</th>
      <th>18278.0</th>
      <th>18280.0</th>
      <th>18281.0</th>
      <th>18282.0</th>
      <th>18283.0</th>
      <th>18287.0</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12347.0</th>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.063022</td>
      <td>0.046130</td>
      <td>0.047795</td>
      <td>0.038484</td>
      <td>0.0</td>
      <td>0.025876</td>
      <td>0.136641</td>
      <td>0.094742</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.029709</td>
      <td>0.052668</td>
      <td>0.0</td>
      <td>0.032844</td>
      <td>0.062318</td>
      <td>0.0</td>
      <td>0.113776</td>
      <td>0.109364</td>
      <td>0.012828</td>
    </tr>
    <tr>
      <th>12348.0</th>
      <td>0.0</td>
      <td>0.063022</td>
      <td>1.000000</td>
      <td>0.024953</td>
      <td>0.051709</td>
      <td>0.027756</td>
      <td>0.0</td>
      <td>0.027995</td>
      <td>0.118262</td>
      <td>0.146427</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.064282</td>
      <td>0.113961</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.170905</td>
      <td>0.083269</td>
    </tr>
    <tr>
      <th>12349.0</th>
      <td>0.0</td>
      <td>0.046130</td>
      <td>0.024953</td>
      <td>1.000000</td>
      <td>0.056773</td>
      <td>0.137137</td>
      <td>0.0</td>
      <td>0.030737</td>
      <td>0.032461</td>
      <td>0.144692</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.105868</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.039014</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.067574</td>
      <td>0.137124</td>
      <td>0.030475</td>
    </tr>
    <tr>
      <th>12350.0</th>
      <td>0.0</td>
      <td>0.047795</td>
      <td>0.051709</td>
      <td>0.056773</td>
      <td>1.000000</td>
      <td>0.031575</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033315</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.044866</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 4338 columns</p>
</div>



## Get the list of customer id sorted by similarity

we can get a list of similar users sorted in descending order by sorting the row values as indexed by CustomerID


```python
user_user_matrix.loc[12350].sort_values(ascending=False)
```




    CustomerID
    12350.0    1.000000
    17935.0    0.183340
    12414.0    0.181902
    12652.0    0.175035
    16754.0    0.171499
                 ...   
    14886.0    0.000000
    14887.0    0.000000
    14888.0    0.000000
    14889.0    0.000000
    18287.0    0.000000
    Name: 12350.0, Length: 4338, dtype: float64



Get the list of items bought by this customer


```python
def get_bought_items(user_item_m, customer_id):
    return set(user_item_m.loc[customer_id].iloc[user_item_m.loc[customer_id].to_numpy().nonzero()].index)
```


```python
items_bought = get_bought_items(user_item_matrix, 12350)
items_bought
```




    {'20615',
     '20652',
     '21171',
     '21832',
     '21864',
     '21866',
     '21908',
     '21915',
     '22348',
     '22412',
     '22551',
     '22557',
     '22620',
     '79066K',
     '79191C',
     '84086C',
     'POST'}



## Get the list of items to recommend to the user

First we need to find out which user is the most similar to the one we are comparing.  Then we compare the set the items our user has bought and the items that the similar user has bought and get the difference in content.  Lastly we try to find the description of the item along with the item stock code to return.


```python
def get_items_to_recommend_user(main_df, user_user_m, user_item_m, user_id):
  most_similar_user = user_user_m.loc[user_id].sort_values(ascending=False).reset_index().iloc[1, 0]
  items_bought_by_user_a = get_bought_items(user_item_m, user_id)
  items_bought_by_user_b = get_bought_items(user_item_m, most_similar_user)
  items_to_recommend_to_a = items_bought_by_user_b - items_bought_by_user_a
  items_description = main_df.loc[main_df['StockCode'].isin(items_to_recommend_to_a), ['StockCode', 'Description']].drop_duplicates().set_index('StockCode')
  return items_description
```


```python
get_items_to_recommend_user(df, user_user_matrix, user_item_matrix, 12358.0)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Description</th>
    </tr>
    <tr>
      <th>StockCode</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85015</th>
      <td>SET OF 12  VINTAGE POSTCARD SET</td>
    </tr>
    <tr>
      <th>16008</th>
      <td>SMALL FOLDING SCISSOR(POINTED EDGE)</td>
    </tr>
  </tbody>
</table>
</div>




```python
most_similar_user = user_user_matrix.loc[12358.0].sort_values(ascending=False).reset_index().iloc[1, 0]
most_similar_user
```




    18240.0




```python
a = get_bought_items(user_item_matrix, 12358.0)
a
```




    {'15056BL',
     '15056N',
     '15056P',
     '15060B',
     '20679',
     '21232',
     '22059',
     '22063',
     '22646',
     '37447',
     '37449',
     '48185',
     'POST'}




```python
b = get_bought_items(user_item_matrix, 18240.0)
b
```




    {'15056BL', '15056N', '15056P', '16008', '20679', '85015'}




```python
b - a
```




    {'16008', '85015'}


