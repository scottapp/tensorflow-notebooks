import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode_categorical_cols(df, cols, enc=None, drop=True):
    if not enc:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(df[cols])

    arry = enc.transform(df[cols]).toarray()
    cat_columns = enc.get_feature_names_out(cols)
    tmp_df = pd.DataFrame(arry, columns=cat_columns)
    df = pd.concat([df, tmp_df], axis=1)

    """
    col_count = len(df.columns)
    df[cat_columns] = arry
    assert len(df.columns) == (col_count + len(cat_columns)), 'error col count, %s' % len(df.columns)
    """

    if drop:
        df.drop(cols, axis=1, inplace=True)
    return enc, df


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
