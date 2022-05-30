from math import pi
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from yellowbrick.cluster import SilhouetteVisualizer

#import lightgbm as lgb
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.offline as pyo

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, norm, probplot
sns.set_style('darkgrid')

import utils as utils


def kmeans_plot_cluster_silscore(dataset, start=2, end=11):
    '''
    Calculate the optimal number of kmeans

    INPUT:
        dataset : dataframe. Dataset for k-means to fit
        start : int. Starting range of kmeans to test
        end : int. Ending range of kmeans to test
    OUTPUT:
        Values and line plot of Silhouette Score.
    '''

    # Create empty lists to store values for plotting graphs
    cluster_list = []
    km_sil_score = []

    # Create a for loop to find optimal n_clusters
    for n_clusters in range(start, end):

        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(dataset)

        silhouette_avg = round(silhouette_score(dataset, labels, random_state=42), 3)

        # Append score to lists
        km_sil_score.append(silhouette_avg)
        cluster_list.append(n_clusters)

        print("No. Clusters: {}, Silhouette Score: {}, Change from Previous Cluster: {}".format(
            n_clusters,
            silhouette_avg,
            (km_sil_score[n_clusters - start] - km_sil_score[n_clusters - start - 1]).round(3)))

        if n_clusters == end - 1:
            plt.figure(figsize=(6.47, 3))

            plt.title('Silhouette Score')
            sns.pointplot(x=cluster_list, y=km_sil_score)
            plt.savefig('silhouette_score.png', format='png', dpi=1000)
            plt.tight_layout()
            plt.show()


def save_last_year_data():
    #df = pd.read_excel('data/Online Retail.xlsx')
    #pd.to_pickle(df, 'data/all_data.pkl')

    df = pd.read_pickle('data/all_data.pkl')
    df.dropna(axis=0, inplace=True, how='any')
    end_date = df['InvoiceDate'].max()
    start_date = end_date - pd.to_timedelta(364, unit='d')
    df_rfm = df[(df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] <= end_date)]
    df_rfm.to_pickle('data/last_year_data.pkl')


def kmeans(df, clusters_number):
    '''
    Implement k-means clustering on dataset

    INPUT:
        dataset : dataframe. Dataset for k-means to fit.
        clusters_number : int. Number of clusters to form.
        end : int. Ending range of kmeans to test.
    OUTPUT:
        Cluster results and t-SNE visualisation of clusters.
    '''

    kmeans = KMeans(n_clusters=clusters_number, random_state=42)
    kmeans.fit(df)

    # Extract cluster labels
    cluster_labels = kmeans.labels_

    # Create a cluster label column in original dataset
    df_new = df.assign(Cluster=cluster_labels)

    # Initialise TSNE
    model = TSNE(random_state=1)
    transformed = model.fit_transform(df)

    # Plot t-SNE
    plt.title('Flattened Graph of {} Clusters'.format(clusters_number))
    sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=cluster_labels, style=cluster_labels, palette="Set1")

    return df_new, cluster_labels


def gen_snake_plot(df, df_norm, labels):
    df_normalized = pd.DataFrame(df_norm, columns=['Recency', 'Frequency', 'MonetarySum'])
    df_normalized['ID'] = df.index
    df_normalized['Cluster'] = labels
    df_nor_melt = pd.melt(df_normalized.reset_index(),
                          id_vars=['ID', 'Cluster'],
                          value_vars=['Recency', 'Frequency', 'MonetarySum'],
                          var_name='Attribute',
                          value_name='Value')
    df_nor_melt.head()

    plt.figure(figsize=(8, 4))
    plt.title('RFM Snake Plot')
    sns.lineplot('Attribute', 'Value', hue='Cluster', data=df_nor_melt)


if __name__ == '__main__':
    """
    cols = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
    #df = pd.read_pickle('data/last_year_data.pkl')
    #df = pd.read_pickle('data/all_data.pkl')
    save_last_year_data(df)
    assert False    
    """

    df_rfm = pd.read_pickle('data/last_year_data.pkl')
    df_rfm['TotalSum'] = df_rfm['Quantity'] * df_rfm['UnitPrice']
    df_rfm = df_rfm.drop(df_rfm.query("Quantity < 0").index)
    end_date = df_rfm['InvoiceDate'].max()
    snapshot_date = end_date + timedelta(days=1)
    df_rfm = df_rfm.groupby(['CustomerID']).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalSum': 'sum'})
    df_rfm.rename(columns={'InvoiceDate': 'Recency',
                           'InvoiceNo': 'Frequency',
                           'TotalSum': 'MonetarySum'},
                  inplace=True)

    df_rfm = df_rfm.drop(df_rfm[df_rfm.MonetarySum == 0].index)
    df_cleaned = df_rfm.copy()

    f, axes = plt.subplots(2, 3, figsize=(12, 5))
    sns.distplot(df_rfm['Recency'], fit=norm, ax=axes[0, 0])
    sns.distplot(df_rfm['Frequency'], fit=norm, ax=axes[0, 1])
    sns.distplot(df_rfm['MonetarySum'], fit=norm, ax=axes[0, 2])

    df_rfm['Recency'] = np.log(df_rfm['Recency'])
    df_rfm['Frequency'] = np.log(df_rfm['Frequency'])
    df_rfm['MonetarySum'] = np.log(df_rfm['MonetarySum'])
    assert len(df_rfm.index[np.isinf(df_rfm).any(1)]) == 0, 'inf was found in data'
    print(df_rfm.describe())

    sns.distplot(df_rfm['Recency'], fit=norm, ax=axes[1, 0])
    sns.distplot(df_rfm['Frequency'], fit=norm, ax=axes[1, 1])
    sns.distplot(df_rfm['MonetarySum'], fit=norm, ax=axes[1, 2])
    plt.show()

    df_rfm_norm = (df_rfm - df_rfm.mean()) / df_rfm.std()
    print(df_rfm.describe())

    kmeans_plot_cluster_silscore(df_rfm_norm)

    # plot silhouette cluster score map
    f, axes = plt.subplots(1, 4, figsize=(30, 10))
    for n in [3, 4, 5, 6]:
        model = KMeans(n, random_state=42)
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick', ax=axes[n-3])
        visualizer.fit(df_rfm_norm)
    visualizer.show()


    # for clusters = 3
    df_new, labels = kmeans(df_rfm_norm, 3)

    # check the stats in each cluster
    df_cleaned['Cluster'] = labels
    df = df_cleaned.groupby('Cluster').agg({'Recency': 'mean',
                                            'Frequency': 'mean',
                                            'MonetarySum': ['mean', 'count']}).round(2)
    print(df.head())

    # Calculate average RFM values for each cluster
    cluster_avg = df_cleaned.groupby('Cluster').mean()
    population_avg = df_cleaned.drop('Cluster', axis=1).mean()
    relative_importance = cluster_avg / population_avg - 1
    plt.figure(figsize=(8, 4))
    plt.title('Relative importance of attributes')
    sns.heatmap(data=relative_importance, annot=True, fmt='.2f', cmap='RdYlGn')

    gen_snake_plot(df_cleaned, df_rfm_norm, labels)

    plt.show()
