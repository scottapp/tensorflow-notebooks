## Fama French 3 Factor Model


```python
import sys
sys.path = ['/app/src'] + sys.path

import io
# Pandas to read csv file and other things
import pandas as pd
# Datareader to download price data from Yahoo Finance
import pandas_datareader as webe
# Statsmodels to run our multiple regression model
import statsmodels.api as smf
# To download the Fama French data from the web
import urllib.request
# To unzip the ZipFile 
import zipfile
import utils
```


```python
def get_fama_french():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    urllib.request.urlretrieve(url, 'fama_french.zip')
    zip_file = zipfile.ZipFile('fama_french.zip', 'r')
    zip_file.extractall()
    zip_file.close()
    
    ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows = 3, index_col = 0)
    ff_row = ff_factors.isnull().any(1).to_numpy().nonzero()[0][0]
    
    ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows = 3, nrows = ff_row, index_col = 0)
    
    ff_factors.index = pd.to_datetime(ff_factors.index, format= '%Y%m')
    
    ff_factors.index = ff_factors.index + pd.offsets.MonthEnd()
    
    ff_factors = ff_factors.apply(lambda x: x/ 100)
    return ff_factors
```

## Load Fama French Data

First we need to download the latest factor data from the following url:

https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip

Download and unzip the csv file in the current directory, then we will convert the csv data into a pandas dataframe that we can use later.  The columns in the data file represent the Mkt excess, SMB, HML factors from which we will perform regression against with later. 



```python
ff_data = get_fama_french()
print(ff_data.tail())
```

                Mkt-RF     SMB     HML      RF
    2021-10-31  0.0665 -0.0228 -0.0044  0.0000
    2021-11-30 -0.0155 -0.0135 -0.0053  0.0000
    2021-12-31  0.0310 -0.0157  0.0323  0.0001
    2022-01-31 -0.0624 -0.0587  0.1279  0.0000
    2022-02-28 -0.0229  0.0219  0.0312  0.0000


another way of parse the factor data file


```python
def load_ff_data():
    with open('data/F-F_Research_Data_Factors.csv', 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = lines[3:1151]
    data = io.StringIO(''.join(lines))
    df = pd.read_csv(data)
    df.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df.index = df.index + pd.offsets.MonthEnd()
    df = df.apply(lambda x: x / 100)
    return df
        
ff_data = load_ff_data()
print(ff_data.tail())
```

                Mkt-RF     SMB     HML      RF
    Date                                      
    2021-09-30 -0.0437  0.0080  0.0509  0.0000
    2021-10-31  0.0665 -0.0228 -0.0044  0.0000
    2021-11-30 -0.0155 -0.0135 -0.0053  0.0000
    2021-12-31  0.0310 -0.0157  0.0323  0.0001
    2022-01-31 -0.0623 -0.0588  0.1278  0.0000


load the prices data of symbol FCNTX for analysis


```python
ff_last = ff_data.index[ff_data.shape[0] - 1].date()
price_data = utils.load_pickle('data/FCNTX_prices.pkl')
price_data = price_data['Adj Close']
price_data = price_data.loc[:ff_last]
print(price_data.tail())
```

    Date
    2022-02-22    15.80
    2022-02-23    15.49
    2022-02-24    15.87
    2022-02-25    16.18
    2022-02-28    16.17
    Name: Adj Close, dtype: float64



```python
def get_return_data(price_data, period = "M"):
    price = price_data.resample(period).last()
    ret_data = price.pct_change()[1:]
    ret_data = pd.DataFrame(ret_data)
    ret_data.columns = ['portfolio']
    return ret_data
    
ret_data = get_return_data(price_data, "M")
print(ret_data.tail())
```

                portfolio
    Date                 
    2021-10-31   0.069570
    2021-11-30  -0.000993
    2021-12-31   0.013778
    2022-01-31  -0.082090
    2022-02-28  -0.048706



```python
# Merging the data
all_data = pd.merge(pd.DataFrame(ret_data), ff_data, how = 'inner', left_index= True, right_index= True)
print(len(all_data))
# Rename the columns
all_data.rename(columns={"Mkt-RF":"mkt_excess"}, inplace=True)
# Calculate the excess returns
all_data['port_excess'] = all_data['portfolio'] - all_data['RF']
print(all_data.head())
print(all_data.tail())
```

    505
                portfolio  mkt_excess     SMB     HML      RF  port_excess
    1980-02-29  -0.022948     -0.0122 -0.0185  0.0061  0.0089    -0.031848
    1980-03-31  -0.089431     -0.1290 -0.0664 -0.0101  0.0121    -0.101531
    1980-04-30   0.017857      0.0397  0.0105  0.0108  0.0126     0.005257
    1980-05-31   0.070175      0.0526  0.0213  0.0038  0.0081     0.062075
    1980-06-30   0.040073      0.0306  0.0166 -0.0076  0.0061     0.033973
                portfolio  mkt_excess     SMB     HML      RF  port_excess
    2021-10-31   0.069570      0.0665 -0.0228 -0.0044  0.0000     0.069570
    2021-11-30  -0.000993     -0.0155 -0.0135 -0.0053  0.0000    -0.000993
    2021-12-31   0.013778      0.0310 -0.0157  0.0323  0.0001     0.013678
    2022-01-31  -0.082090     -0.0624 -0.0587  0.1279  0.0000    -0.082090
    2022-02-28  -0.048706     -0.0229  0.0219  0.0312  0.0000    -0.048706



```python
model = smf.formula.ols(formula = "port_excess ~ mkt_excess + SMB + HML", data=all_data).fit()
print(model.params)
```

    Intercept     0.001305
    mkt_excess    0.885457
    SMB           0.021331
    HML          -0.121153
    dtype: float64



```python

```
