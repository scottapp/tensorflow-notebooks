import uuid
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

from scipy.stats import skew, norm, probplot

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from utils import one_hot_encode_categorical_cols
from utils import has_null, null_count, plot_simple, plot_values_distribution
from utils import get_mean, get_mode, get_col_names, get_col_dtype


def create_model(info, input_len):
    info['model'] = '1024_mae'
    model = Sequential()
    model.add(Dense(1024, input_dim=input_len, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    #model.add(tf.keras.layers.Lambda(lambda x: (x * 0.4) + 12.02))
    # Compile model
    #model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mae', metrics=['msle'])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
    #model.compile(optimizer='adam', loss='mse')
    #model.compile(optimizer='adam', loss='mae')
    #model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    return model


def remove_outliers(info, df):
    info['outliers'] = 'iso'

    clf = IsolationForest(max_samples=100, random_state=42)
    clf.fit(df)
    y_noano = clf.predict(df)
    y_noano = pd.DataFrame(y_noano, columns=['Top'])
    #y_noano[y_noano['Top'] == 1].index.values
    df = df.iloc[y_noano[y_noano['Top'] == 1].index.values]
    print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
    print("Number of rows without outliers:", df.shape[0])
    return df


def calc_extra_features(info, df):
    info['features'] = 'extra'
    df['Total_Square_Feet'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['Total_Bath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df['Total_Porch_Area'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df['SqFtPerRoom'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + df['FullBath'] + df['HalfBath'] + df['KitchenAbvGr'])


def get_null_cols(df):
    out = list()
    for col in df.columns:
        if df[col].isnull().values.any():
            out.append(col)
    return out


def clean_data_01(info, df, df_test):
    info['clean'] = 'clean_data_01'

    full = pd.concat([df, df_test])

    #cols_to_remove = ['Id', 'Alley', 'GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature']
    cols_to_remove = ['Alley', 'GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature']

    full.drop(cols_to_remove, axis=1, inplace=True)
    #print(full.shape)

    full['LotFrontage'].fillna(full.groupby('1stFlrSF')['LotFrontage'].transform('mean'), inplace=True)
    full['LotFrontage'].interpolate(method='linear', inplace=True)
    full['MasVnrArea'].fillna(full.groupby('MasVnrType')['MasVnrArea'].transform('mean'), inplace=True)
    full['MasVnrArea'].interpolate(method='linear', inplace=True)

    cat_cols = full.select_dtypes(include=['object']).columns
    for col in cat_cols:
        full[col].fillna(full[col].mode()[0], inplace=True)

    full.loc[((full['BsmtExposure'].isnull()) & (full['BsmtFinType1'].notnull())), 'BsmtExposure'] = 'No'
    full.loc[((full['BsmtExposure'].isnull()) & (full['BsmtFinType1'].notnull())), 'BsmtExposure'] = 'No'

    # We impute missing basement condition with "mean" value of Typical.
    full.loc[((full['BsmtCond'].isnull()) & (full['BsmtFinType1'].notnull())), 'BsmtCond'] = 'TA'

    # We impute unfinished basement quality with "mean" value of Typical.
    full.loc[((full['BsmtQual'].isnull()) & (full['BsmtFinType1'].notnull())), 'BsmtQual'] = 'TA'

    # 'BsmtFullBath', 'BsmtHalfBath'
    full.loc[full['BsmtHalfBath'].isnull(), 'BsmtHalfBath'] = 0
    full.loc[full['BsmtFullBath'].isnull(), 'BsmtFullBath'] = 0

    for col in ['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']:
        full.loc[full[col].isnull(), col] = 0

    full['GarageCars'].fillna(2, inplace=True)
    full['GarageArea'].fillna(get_mean(full, "GarageType == 'Detchd'", 'GarageArea'), inplace=True)

    null_cols = get_null_cols(full)
    assert len(null_cols) == 1 and null_cols[0] == 'SalePrice', '%s columns with null values' % len(null_cols)
    return full[:len(df)], full[len(df):].drop('SalePrice', axis=1)


def clean_data_02(info, df, df_test):
    info['clean'] = 'clean_data_02'

    #y = df['SalePrice']
    #df = df.drop('SalePrice', axis=1)
    df = df.set_index('Id')
    df_test = df_test.set_index('Id')

    full = pd.concat([df, df_test], axis=0)

    # categorical_cols = list(df.select_dtypes(include=['object']).columns)
    # numerical_cols = list(df.select_dtypes(include=['int', 'float']).columns)

    df_test.loc[(df_test['Neighborhood'] == 'IDOTRR') & (df_test['MSZoning'].isnull()), 'MSZoning'] = 'RM'
    df_test.loc[(df_test['Neighborhood'] == 'Mitchel') & (df_test['MSZoning'].isnull()), 'MSZoning'] = 'RL'

    # impute LotFrontage
    area_vs_frontage = LinearRegression()
    data = full[(~full['LotFrontage'].isnull()) & (full['LotFrontage'] <= 150) & (full['LotArea'] <= 20000)]
    area_vs_frontage_X = data['LotArea'].values.reshape(-1, 1)
    area_vs_frontage_y = data['LotFrontage'].values
    area_vs_frontage.fit(area_vs_frontage_X, area_vs_frontage_y)
    for table in [df, df_test]:
        table['LotFrontage'].fillna(area_vs_frontage.intercept_ + table['LotArea'] * area_vs_frontage.coef_[0],
                                    inplace=True)

    # Alley
    for table in [df, df_test]:
        table['Alley'].fillna("None", inplace=True)

    # Utilities
    df_test['Utilities'].fillna("AllPub", inplace=True)

    # Exterior
    df_test['Exterior1st'] = df_test['Exterior1st'].fillna(full['Exterior1st'].mode()[0])
    df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(full['Exterior2nd'].mode()[0])

    df_test.loc[2611, 'MasVnrType'] = 'BrkFace'
    df_test['MasVnrType'] = df_test['MasVnrType'].fillna(full['MasVnrType'].mode()[0])
    df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(0)

    df['MasVnrType'] = df['MasVnrType'].fillna(full['MasVnrType'].mode()[0])
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    # We assume missing basement exposure of unfinished basement is "No".
    df.loc[((df['BsmtExposure'].isnull()) & (df['BsmtFinType1'].notnull())), 'BsmtExposure'] = 'No'
    df_test.loc[((df_test['BsmtExposure'].isnull()) & (df_test['BsmtFinType1'].notnull())), 'BsmtExposure'] = 'No'

    # We impute missing basement condition with "mean" value of Typical.
    df_test.loc[((df_test['BsmtCond'].isnull()) & (df_test['BsmtFinType1'].notnull())), 'BsmtCond'] = 'TA'

    # We impute unfinished basement quality with "mean" value of Typical.
    df_test.loc[((df_test['BsmtQual'].isnull()) & (df_test['BsmtFinType1'].notnull())), 'BsmtQual'] = 'TA'

    for col in ['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']:
        df_test.loc[2121, col] = 0

    for col in ['BsmtFullBath', 'BsmtHalfBath']:
        df_test.loc[2121, col] = 0
        df_test.loc[2189, col] = 0

    cols = ['BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1']
    for table in [df, df_test]:
        table[cols] = table[cols].fillna("None")

    df['Electrical'].fillna('SBrkr', inplace=True)
    df_test['Functional'].fillna('Typ', inplace=True)
    df_test['KitchenQual'].fillna('TA', inplace=True)

    df_test['GarageCars'].fillna(2, inplace=True)
    df_test['GarageArea'].fillna(get_mean(full, "GarageType == 'Detchd'", 'GarageArea'), inplace=True)

    df_test['SaleType'].fillna('WD', inplace=True)

    df['GarageYrBlt'].fillna(0, inplace=True)
    df_test['GarageYrBlt'].fillna(0, inplace=True)

    """
    OverallQualCond = {
        10: "Very Excellent",
        9: "Excellent",
        8: "Very Good",
        7: "Good",
        6: "Above Average",
        5: "Average",
        4: "Below Average",
        3: "Fair",
        2: "Poor",
        1: "Very Poor",
    }
    df.replace({"OverallQual": OverallQualCond}, inplace=True)
    df_test.replace({"OverallQual": OverallQualCond}, inplace=True)
    """

    # fill all other null with None
    df.fillna("None", inplace=True)
    df_test.fillna("None", inplace=True)

    null_cols = get_null_cols(df)
    assert len(null_cols) == 0, 'column has nulls'

    null_cols = get_null_cols(df_test)
    assert len(null_cols) == 0, 'column has nulls'

    assert 'SalePrice' not in df_test.columns
    df.reset_index(inplace=True)
    df_test.reset_index(inplace=True)

    return df, df_test


def clean_simple(info, df, df_test):
    info['clean'] = 'clean_simple'
    full = pd.concat([df, df_test])

    # remove columns
    cols_to_remove = ['Alley', 'GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature']
    #cols_to_remove = ['Id', 'Alley', 'GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature']
    full.drop(cols_to_remove, axis=1, inplace=True)

    # fill mode
    cat_cols = list(full.select_dtypes(include=['object']).columns)
    for col in cat_cols:
        full[col].fillna(full[col].mode()[0], inplace=True)

    # fill 0 except SalePrice
    num_cols = list(full.select_dtypes(include=['int', 'float']).columns)
    num_cols.remove('SalePrice')
    for col in num_cols:
        full[col].fillna(0, inplace=True)

    return full[:len(df)], full[len(df):].drop('SalePrice', axis=1)


def encode_cat_cols(info, df, df_test):
    info['encode'] = '1'
    full = pd.concat([df, df_test])
    cat_cols = list(full.select_dtypes(include=['object']).columns)
    encoder, full = one_hot_encode_categorical_cols(full, cat_cols)
    return encoder, full[:len(df)], full[len(df):].drop('SalePrice', axis=1)


def scale_num_cols(info, df, scaler_type='minmax', scaler=None):
    cat_cols = list(df.select_dtypes(include=['object']).columns)
    num_cols = list(df.select_dtypes(include=['number']).columns)
    add_back = list()
    if 'SalePrice' in num_cols:
        add_back.append('SalePrice')
        num_cols.remove('SalePrice')
    if not scaler:
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
            info['scaler'] = 'minmax'
        else:
            scaler = StandardScaler()
            info['scaler'] = 'standard'
        df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
    else:
        df_scaled = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols)
    if add_back:
        cat_cols = cat_cols + add_back
    result = pd.merge(df[cat_cols], df_scaled, left_index=True, right_index=True)
    return scaler, result


def fit_validate_model(info, model, X_train, y_train, trial=False):
    info['fit_val'] = 'val_split_10'

    if trial:
        epoch_count = 3
    else:
        epoch_count = 1000

    # val_loss, val_msle
    monitor = 'val_loss'
    early_stop = EarlyStopping(monitor=monitor,
                               mode='min',
                               verbose=1,
                               patience=20)
    history = model.fit(x=X_train,
                        y=y_train,
                        validation_split=0.1,
                        batch_size=128,
                        epochs=epoch_count,
                        callbacks=[early_stop])
    return history


def save_pipeline_info(info, submission=None):
    rnd_str = str(uuid.uuid4())
    info['id'] = rnd_str
    with open('output/%s_%s.json' % (info['clean'], rnd_str), 'w', encoding='utf8') as f:
        f.write(json.dumps(info, indent=2))
    submission.to_csv('output/%s_%s.csv' % (info['clean'], rnd_str), index=False)


def pipeline_simple(options):
    trial = options['trial']
    scaler_type = options['scaler_type']

    df = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    cat_cols = list(df.select_dtypes(include=['object']).columns)
    num_cols = list(df.select_dtypes(include=['number']).columns)
    assert (len(cat_cols) + len(num_cols)) == len(df.columns), 'error columns count'
    """
    print(len(cat_cols))
    print(len(num_cols))
    print(len(df.columns))
    """

    info = dict()
    df, df_test = clean_simple(info, df, df_test)
    print(df.shape)
    print(df_test.shape)

    ids = df_test['Id'].values

    scaler, df = scale_num_cols(info, df, scaler_type=scaler_type)
    scaler, df_test = scale_num_cols(info, df_test, scaler=scaler)
    print(df.shape)
    print(df_test.shape)

    encoder, df, df_test = encode_cat_cols(info, df, df_test)
    print(df.shape)
    print(df_test.shape)

    df = remove_outliers(info, df)

    assert df.isnull().values.any() == False, 'df still has null values'
    assert df_test.isnull().values.any() == False, 'df_test still has null values'
    assert 'SalePrice' in df.columns, 'missing SalePrice column'

    y_train = df['SalePrice'].values
    X_train = df.drop(['Id', 'SalePrice'], axis=1).values
    X_test = df_test.drop('Id', axis=1).values
    assert X_train.shape[1] == X_test.shape[1], 'error columns count'

    model = create_model(info, X_train.shape[1])
    history = fit_validate_model(info, model, X_train, y_train, trial=trial)
    epoch_count = len(history.epoch)
    info['epoch_count'] = epoch_count

    model = create_model(info, X_train.shape[1])
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=128,
                        epochs=epoch_count)

    eval = model.evaluate(X_train, y_train)
    print(eval)

    y_test_pred = model.predict(X_test)
    assert y_test_pred.shape[0] == ids.shape[0], 'error shape, %s, %s' % (y_test_pred.shape[0], ids.shape[0])

    y_pred = model.predict(X_train)
    print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(y_train, y_pred)))
    print('Mean Squared Error: {:.2f}'.format(metrics.mean_squared_error(y_train, y_pred)))
    print('Root Mean Squared Error: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_train, y_pred))))
    print('Variance score is: {:.2f}'.format(metrics.explained_variance_score(y_train, y_pred)))
    info['var_score'] = '{:.5f}'.format(metrics.explained_variance_score(y_train, y_pred))

    submission = pd.DataFrame(y_test_pred, columns=['SalePrice'])
    submission['Id'] = ids
    save_pipeline_info(info, submission)
    print(info)


def pipeline(options):
    trial = options['trial']
    scaler_type = options['scaler_type']

    df = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    cat_cols = list(df.select_dtypes(include=['object']).columns)
    num_cols = list(df.select_dtypes(include=['number']).columns)
    assert (len(cat_cols) + len(num_cols)) == len(df.columns), 'error columns count'

    clean_func = options['clean']
    info = dict()

    if clean_func == 'clean_data_01':
        df, df_test = clean_data_01(info, df, df_test)
    elif clean_func == 'clean_data_02':
        df, df_test = clean_data_02(info, df, df_test)
    else:
        assert False

    print(df.shape)
    print(df_test.shape)

    calc_extra_features(info, df)
    calc_extra_features(info, df_test)

    ids = df_test['Id'].values

    scaler, df = scale_num_cols(info, df, scaler_type=scaler_type)
    scaler, df_test = scale_num_cols(info, df_test, scaler=scaler)
    print(df.shape)
    print(df_test.shape)

    encoder, df, df_test = encode_cat_cols(info, df, df_test)
    print(df.shape)
    print(df_test.shape)

    df = remove_outliers(info, df)

    assert df.isnull().values.any() == False, 'df still has null values'
    assert df_test.isnull().values.any() == False, 'df_test still has null values'
    assert 'SalePrice' in df.columns, 'missing SalePrice column'

    y_train = df['SalePrice'].values
    X_train = df.drop(['Id', 'SalePrice'], axis=1).values
    X_test = df_test.drop('Id', axis=1).values

    assert X_train.shape[1] == X_test.shape[1], 'error columns count'

    model = create_model(info, X_train.shape[1])
    history = fit_validate_model(info, model, X_train, y_train, trial=trial)
    epoch_count = len(history.epoch)
    info['epoch_count'] = epoch_count

    model = create_model(info, X_train.shape[1])
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=128,
                        epochs=epoch_count)

    eval = model.evaluate(X_train, y_train)
    print(eval)

    y_test_pred = model.predict(X_test)
    assert y_test_pred.shape[0] == ids.shape[0], 'error shape, %s, %s' % (y_test_pred.shape[0], ids.shape[0])

    y_pred = model.predict(X_train)
    print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(y_train, y_pred)))
    print('Mean Squared Error: {:.2f}'.format(metrics.mean_squared_error(y_train, y_pred)))
    print('Root Mean Squared Error: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_train, y_pred))))
    print('Variance score is: {:.2f}'.format(metrics.explained_variance_score(y_train, y_pred)))
    info['var_score'] = '{:.5f}'.format(metrics.explained_variance_score(y_train, y_pred))

    submission = pd.DataFrame(y_test_pred, columns=['SalePrice'])
    submission['Id'] = ids
    save_pipeline_info(info, submission)
    print(info)


if __name__ == '__main__':

    trial = False
    #scaler_types = ['minmax', 'standard']
    scaler_types = ['standard']

    for scaler_type in scaler_types:
        options = dict()

        options['scaler_type'] = scaler_type
        options['trial'] = trial

        """
        #options['monitor'] = 'val_loss'
        #options['monitor'] = 'val_msle'
        pipeline_simple(options)

        options['clean'] = 'clean_data_01'
        pipeline(options)
        """

        options['clean'] = 'clean_data_02'
        pipeline(options)
