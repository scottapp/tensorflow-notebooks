import pandas as pd
import numpy as np
import seaborn as sns

"""
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping # Early Stopping Callback in the NN

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
"""
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def has_null(df, col):
    return df[col].isnull().any()


def one_hot_encode_categorical_cols(df, cols, enc=None, drop=True):
    if not enc:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(df[cols])

    arry = enc.transform(df[cols]).toarray()
    cat_columns = enc.get_feature_names_out(cols)
    tmp_df = pd.DataFrame(arry, columns=cat_columns)

    df.reset_index(inplace=True, drop=True)
    df = pd.concat([df, tmp_df], axis=1)

    """
    col_count = len(df.columns)
    df[cat_columns] = arry
    assert len(df.columns) == (col_count + len(cat_columns)), 'error col count, %s' % len(df.columns)
    """

    if drop:
        df.drop(cols, axis=1, inplace=True)
    return enc, df


def get_mean(df, query, col_name):
    return df.query(query)[col_name].mean()


def get_mode(df, query, col_name):
    return df.query(query)[col_name].mode()


def get_col_names(df, match_str):
    return list(df.columns[df.columns.str.contains(match_str)])


def get_col_dtype(df, cols, show_all=False):
    output = list()
    for col in cols:
        if show_all:
            output.append([col, df[col].dtype, df[col].isnull().sum()])
        else:
            if df[col].isnull().values.any():
                output.append([col, df[col].dtype, df[col].isnull().sum()])

    df = pd.DataFrame(output, columns=['feature', 'dtype', 'null count']).sort_values(by='null count', ascending=False)
    return df


def create_col_names(df_list):
    output = list()
    for i in range(0, len(df_list)):
        output.append('df_%s_null' % i)
        output.append('df_%s_mode' % i)
        output.append('df_%s_mean' % i)
        output.append('df_%s_nuniq' % i)
    return output


def null_count(df_list, cols):
    assert type(df_list) == list, 'error input dataframe list, %s' % type(df_list)
    assert type(cols) == list, 'error input col name list, %s' % type(cols)

    null_rows = list()
    for col in cols:
        has_null = False
        col_dtype = None
        # append feature name first
        null_row = [col]
        for df in df_list:
            if not col_dtype:
                col_dtype = df[col].dtype

            # append sum for this df
            null_sum = df[col].isnull().sum()
            null_row.append(null_sum)

            # append mode
            mode_val = df[col].mode()[0]
            null_row.append(mode_val)

            # append mean
            if df[col].dtype == 'float':
                null_row.append(df[col].mean())
            else:
                null_row.append('')

            # append nunique
            if df[col].dtype == 'int' or df[col].dtype == 'object':
                null_row.append(df[col].nunique())
            else:
                null_row.append('')

            if null_sum > 0:
                has_null = True

        if has_null:
            null_row.append(col_dtype)
            null_rows.append(null_row)

    col_names = create_col_names(df_list)

    df_rows_count = dict()
    count = 0
    for df in df_list:
        df_rows_count['df_%s'%count] = len(df)
        count += 1

    null_df = pd.DataFrame(null_rows, columns=['feature']+col_names+['dtype'])
    null_df['Total Null'] = null_df.sum(axis=1)
    for col in col_names:
        if col.endswith('_null'):
            null_df['%s_pct' % col] = null_df[col] / df_rows_count[col.replace('_null', '')]
    return null_df.sort_values(by='Total Null', ascending=False)


def plot_simple(df, x_name, y_name, rotation='v', show_fit=False):
    x = list(df[x_name])
    y = list(df[y_name])
    # sort y, change x
    y, x = zip(*sorted(zip(y, x)))

    plt.plot(x, y, 'o')
    plt.xlabel(x_name)
    if rotation == 'v':
        plt.xticks(rotation='vertical')
    else:
        plt.xticks(rotation='horizontal')
    plt.ylabel(y_name)
    plt.grid()

    if show_fit:
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--")
        except Exception as ex:
            pass
            #print(ex)
        plt.show()


def plot_values_distribution(df, col_name):
    table = df[col_name].value_counts()
    sns.barplot(x=table.index, y=table.values)
    plt.title("Distribution plot of " + col_name)
    plt.show()


def test_one_hot_encode_categorical_cols():
    data = dict()
    data['id'] = [1, 2, 3]
    data['cat'] = ['A', 'B', 'C']
    df = pd.DataFrame(data)
    enc, df = one_hot_encode_categorical_cols(df, ['cat'])
    assert len(df.columns) == 4
    assert 'cat_A' in df.columns
    assert 'cat_B' in df.columns
    assert 'cat_C' in df.columns
    print(df)


def test_one_hot_encode_categorical_cols_2():
    data = dict()
    data['id'] = [1, 2, 3]
    data['cat1'] = ['A', 'B', 'C']
    data['cat2'] = ['D', 'E', 'F']
    df = pd.DataFrame(data)
    enc, df = one_hot_encode_categorical_cols(df, ['cat1', 'cat2'])
    assert len(df.columns) == 7
    assert 'cat1_A' in df.columns
    assert 'cat1_B' in df.columns
    assert 'cat1_C' in df.columns
    assert 'cat2_D' in df.columns
    assert 'cat2_E' in df.columns
    assert 'cat2_F' in df.columns

    data = dict()
    data['id'] = [1, 2, 3]
    data['cat1'] = ['X', 'Y', 'Z']
    data['cat2'] = ['X', 'Y', 'Z']
    tmp = pd.DataFrame(data)
    arry = enc.transform(tmp[['cat1', 'cat2']]).toarray()
    assert np.sum(arry) == 0

    data = dict()
    data['id'] = [1, 2, 3]
    data['cat1'] = ['A', 'Y', 'Z']
    data['cat2'] = ['X', 'Y', 'Z']
    tmp = pd.DataFrame(data)
    arry = enc.transform(tmp[['cat1', 'cat2']]).toarray()
    assert np.sum(arry) == 1

    data = dict()
    data['id'] = [1, 2, 3]
    data['cat1'] = ['A', 'Y', 'Z']
    data['cat2'] = ['D', 'Y', 'Z']
    tmp = pd.DataFrame(data)
    arry = enc.transform(tmp[['cat1', 'cat2']]).toarray()
    assert np.sum(arry) == 2
