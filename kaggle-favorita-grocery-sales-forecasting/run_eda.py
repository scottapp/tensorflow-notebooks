import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
#import lightgbm as lgb
from datetime import date, timedelta

import matplotlib.pyplot as plt
import seaborn as sns

import utils as utils


if __name__ == '__main__':
    path = './data/'

    test = pd.read_csv(path + "test.csv")
    tmp = utils.basic_info(test)
    print(tmp.head())
    assert test.isnull().values.any() == False, 'df has null values'

    transactions = pd.read_csv(path + "transactions.csv")
    tmp = utils.basic_info(transactions)
    print(tmp.head())
    assert transactions.isnull().values.any() == False, 'df has null values'

    items = pd.read_csv(path + "items.csv")
    tmp = utils.basic_info(items)
    print(tmp.head())
    assert items.isnull().values.any() == False, 'df has null values'

    stores = pd.read_csv(path + "stores.csv")
    tmp = utils.basic_info(stores)
    print(tmp.head())
    assert stores.isnull().values.any() == False, 'df has null values'

    holidays = pd.read_csv(path + "holidays_events.csv")
    tmp = utils.basic_info(holidays)
    print(tmp.head())
    assert holidays.isnull().values.any() == False, 'df has null values'
