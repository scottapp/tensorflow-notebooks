from math import pi
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#import lightgbm as lgb
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.offline as pyo

import matplotlib.pyplot as plt
import seaborn as sns

import utils as utils


def merge_data(purchases, distances, prices):

    output = pd.merge(purchases, prices, how='left', on='product_id')
    output['amount'] = output['quantity'] * output['price']

    output = pd.merge(output, distances, how='left', on=['customer_id', 'shop_id'])
    output['cnt_txn'] = 1
    return output


def preprocess_customer_shop(df, value_col, func, seperate_shop_id=False):
    if func == 'sum':
        suffix = 'total'
    elif func == 'mean':
        suffix = 'avg'
    else:
        suffix = func

    if seperate_shop_id:
        tmp = df
    else:
        tmp = df.copy()
        tmp['shop_id'] = 'all'
    pv = pd.pivot_table(tmp,
                        values=value_col,
                        index='customer_id',
                        columns='shop_id',
                        aggfunc=func)
    pv.fillna(0, inplace=True)

    if seperate_shop_id:
        pv.columns = ['shop_id{}_{}_{}'.format(x, value_col, suffix) for x in pv.columns.values]
    else:
        pv.columns = ['all_shops_{}_{}'.format(value_col, suffix)]

    pv.reset_index(inplace=True)
    return pv


def preprocess(df, value_col, func, shop_id_only=False, total_only=False):
    if func == 'sum':
        suffix = 'total'
    elif func == 'mean':
        suffix = 'avg'
    else:
        suffix = func

    # by shop id
    pv1 = pd.pivot_table(df,
                         values=value_col,
                         index='customer_id',
                         columns='shop_id',
                         aggfunc=func)
    pv1.fillna(0, inplace=True)
    pv1.columns = ['shop_id{}_{}_{}'.format(x, value_col, suffix) for x in pv1.columns.values]
    pv1.reset_index(inplace=True)

    # total by making shop id the same
    tmp = df.copy()
    tmp['shop_id'] = 'all'
    pv2 = pd.pivot_table(tmp,
                         values=value_col,
                         index='customer_id',
                         columns='shop_id',
                         aggfunc=func)
    pv2.fillna(0, inplace=True)
    pv2.columns = ['all_shops_{}_{}'.format(value_col, suffix)]
    pv2.reset_index(inplace=True)

    output = pd.merge(pv1, pv2, on='customer_id', how='left')
    return output


def plot_radar_demo():
    categories = ['Food Quality', 'Food Variety', 'Service Quality', 'Ambience', 'Affordability']
    categories = [*categories, categories[0]]

    restaurant_1 = [4, 4, 5, 4, 3]
    restaurant_2 = [5, 5, 4, 5, 2]
    restaurant_3 = [3, 4, 5, 3, 5]
    restaurant_1 = [*restaurant_1, restaurant_1[0]]
    restaurant_2 = [*restaurant_2, restaurant_2[0]]
    restaurant_3 = [*restaurant_3, restaurant_3[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(r=restaurant_1, theta=categories, name='Restaurant 1'),
            go.Scatterpolar(r=restaurant_2, theta=categories, name='Restaurant 2'),
            go.Scatterpolar(r=restaurant_3, theta=categories, name='Restaurant 3')
        ],
        layout=go.Layout(
            title=go.layout.Title(text='Restaurant comparison'),
            polar={'radialaxis': {'visible': True}},
            showlegend=True
        )
    )

    pyo.plot(fig)


def save_last_year_data():
    df = pd.read_excel('data/Online Retail.xlsx')
    pd.to_pickle(df, 'data/all_data.pkl')
    max_date = df.InvoiceDate.max()
    df = df[df.InvoiceDate > max_date - timedelta(364)]
    df.to_pickle('data/last_year_data.pkl')


if __name__ == '__main__':
    cols = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']

    df = pd.read_pickle('data/last_year_data.pkl')
    df = pd.read_pickle('data/all_data.pkl')

    """
    save_last_year_data(df)
    assert False
    
    df = pd.read_pickle('data/last_year_data.pkl')
    tmp = utils.basic_info(df)
    print(tmp)
    print(df.describe())
    print(df.info())
    """
    df.dropna(axis=0, inplace=True, how='any')

    end_date = df['InvoiceDate'].max()
    start_date = end_date - pd.to_timedelta(364, unit='d')
    df_rfm = df[(df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] <= end_date)]
    df_rfm.to_pickle('data/last_year_data.pkl')
    assert False

    path = './data/'

    """
    purchases = pd.read_csv(path + "supermarket_purchases.csv")
    assert False
    """

    purchases = utils.get_first_chunk(path + "supermarket_purchases.csv", chunksize=100000)
    #print(purchases.head())
    #print(purchases.info())
    tmp = utils.basic_info(purchases)
    assert purchases.isnull().values.any() == False, 'df has null values'

    distances = pd.read_csv(path + "supermarket_distances.csv", delim_whitespace=True)
    tmp = utils.basic_info(distances)
    print(tmp.head())
    assert distances.isnull().values.any() == False, 'df has null values'

    prices = pd.read_csv(path + "supermarket_prices.csv", delim_whitespace=True)
    tmp = utils.basic_info(prices)
    print(tmp.head())
    print(prices.describe())
    assert prices.isnull().values.any() == False, 'df has null values'

    df = merge_data(purchases, distances, prices)
    print(df.describe())
    print(len(df))

    print('number of customers = {}'.format(len(df['customer_id'].unique())))
    print('number of products = {}'.format(len(df['product_id'].unique())))
    print('number of shops = {}'.format(len(df['shop_id'].unique())))

    features = list()
    features.append(preprocess(df, 'amount', 'sum'))
    features.append(preprocess(df, 'amount', 'mean'))

    features.append(preprocess(df, 'quantity', 'sum'))
    features.append(preprocess(df, 'quantity', 'mean'))

    features.append(preprocess(df, 'price', 'mean'))

    features.append(preprocess_customer_shop(df, 'distance', 'mean', False))
    features.append(preprocess_customer_shop(df, 'distance', 'min', False))
    features.append(preprocess_customer_shop(df, 'distance', 'max', False))
    features.append(preprocess_customer_shop(df, 'distance', 'mean', True))

    # how many shops a customer has been
    tmp = preprocess_customer_shop(df, 'cnt_txn', 'sum', True)
    tmp['shops_visited'] = preprocess_customer_shop(df, 'cnt_txn', 'max', True).drop('customer_id', axis=1).sum(axis=1)
    features.append(tmp)

    merged = features[0]
    for tmp_df in features[1:]:
        merged = pd.merge(merged, tmp_df, how='left', on='customer_id')

    for col in merged.columns:
        assert not col.endswith('_y'), 'might have duplicated columns'

    features = merged
    features_norm = (features - features.mean()) / features.std()
    features_norm = features_norm[sorted(features_norm.columns)]
    print(features_norm.head())

    inertia = []
    for k in range(2, 16):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(features_norm)
        score = km.inertia_
        inertia.append(score)

    plt.figure(1, figsize=(10, 6))
    plt.plot(np.arange(2, 16,), inertia, '-', alpha=0.5)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    sil_scores = list()
    for k in range(4, 16):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(features_norm)
        labels = km.labels_
        sil_scores.append(silhouette_score(features_norm, labels, metric='euclidean'))

    plt.figure(1, figsize=(10, 6))
    plt.plot(np.arange(4, 16, ), sil_scores, '-', alpha=0.5)
    plt.show()

    number_clusters = 6
    km = KMeans(n_clusters=number_clusters, random_state=42)
    km.fit(features_norm)

    result = features.copy()
    result['cluster'] = km.predict(features_norm)
    result.cluster.value_counts().plot.bar()
    plt.show()

    pca = PCA(n_components=2)
    pca_feat = pca.fit_transform(features_norm)
    plt.scatter(x=pca_feat[:, 0], y=pca_feat[:, 1], c=result['cluster'], alpha=0.5)
    plt.figure(figsize=(20, 20))
    plt.show()

    avg_cluster = result.groupby(by='cluster').mean()
    avg_cluster.reset_index(inplace=True)

    shop_amount = avg_cluster[['shop_id1_amount_total',
                               'shop_id2_amount_total',
                               'shop_id3_amount_total',
                               'shop_id4_amount_total',
                               'shop_id5_amount_total',
                               'cluster']]
    shop_distance = avg_cluster[['shop_id1_distance_avg',
                                 'shop_id2_distance_avg',
                                 'shop_id3_distance_avg',
                                 'shop_id4_distance_avg',
                                 'shop_id5_distance_avg',
                                 'cluster']]
    df_melted = pd.melt(shop_amount, id_vars='cluster', var_name='shop_id', value_name='total')
    print(df.head(10))
    sns.barplot(data=df_melted, x='cluster', y='total', hue='shop_id')
    plt.show()

    plot_radar_demo()

    print('done')
