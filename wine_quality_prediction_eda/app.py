import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics

from ml_data import utils


def independence_test(sample_a, sample_b, col):
    statistic, p_value = stats.ttest_ind(sample_a[col], sample_b[col])
    return p_value


def run_eda():
    # df = pd.read_csv("data/WineQT.csv")
    df = pd.read_csv("data/winequality-red.csv", sep=';')
    df2 = pd.read_csv("data/winequality-white.csv", sep=';')
    df = pd.concat([df, df2])
    assert df.isnull().values.sum() == 0, 'nan value in dataset'

    print(df.head())
    print(df.info())
    print(df.columns)
    print(df.nunique())
    print(df.describe())
    print(df.isnull().values.sum())

    df = df.reset_index(drop=True)

    # univariate distribution plots
    f, axes = plt.subplots(len(df.columns), figsize=(20, 60), sharex=False)
    for i in range(len(df.columns)):
        d = sns.histplot(x=df[df.columns[i]], kde=True, stat='density', ax=axes[i])
        d.set_xlabel(df.columns[i], fontsize=15)
        utils.plot_normal(axes[i], df[df.columns[i]].mean(), df[df.columns[i]].std())

    plt.savefig('01_univariate_distribution.png', bbox_inches='tight')

    plt.figure(figsize=[18, 7])
    sns.heatmap(df.corr(), cmap='Blues', annot=True)
    plt.savefig('02_corr_heapmap.png', bbox_inches='tight')
    plt.show()

    """
    df = df.reset_index(drop=True)
    sns.pairplot(df)
    plt.show()
    """

    print(df.groupby('quality')['alcohol'].describe().T)

    sample_a = df[df['quality'] == 5].reset_index(drop=True)
    sample_b = df[df['quality'] == 6].reset_index(drop=True)
    sample_b = sample_b.sample(2138)
    assert sample_a.shape == sample_b.shape, 'error shapes'
    for col in df.columns:
        if col == 'quality':
            continue
        val = independence_test(sample_a, sample_b, col)
        if val <= 0.05:
            print(f'There is difference in mean values of {col} for groups of quality 5 and 6')
        else:
            print(f'No difference in mean values of {col} for groups of quality 5 and 6')

    diff = scale((np.array(sample_a['alcohol']) - np.array(sample_b['alcohol'])))
    plt.figure(figsize=(12, 6))
    plt.hist(diff)
    plt.title('residual distribution for alcohol')
    plt.savefig('03_residual_distribution_alcohol.png', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 6))
    stats.probplot(diff, plot=plt, dist='norm')
    plt.title('residual Q-Q plot for alcohol')
    plt.savefig('04_residual_qq_plot_alcohol.png', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.kdeplot(sample_a['alcohol'], shade=True)
    sns.kdeplot(sample_b['alcohol'], shade=True)
    plt.legend(['sample_a', 'sample_b'], fontsize=14)
    plt.vlines(x=sample_a['alcohol'].mean(), ymin=0, ymax=1, color='blue', linestyle='--')
    plt.vlines(x=sample_b['alcohol'].mean(), ymin=0, ymax=1, color='red', linestyle='--')
    plt.title('distribution of alcohol for the 2 samples')
    plt.savefig('05_dist_alcohol_2_samples.png', bbox_inches='tight')
    plt.show()

    col = 'residual sugar'
    plt.figure(figsize=(12, 6))
    sns.kdeplot(sample_a[col], shade=True)
    sns.kdeplot(sample_b[col], shade=True)
    plt.legend(['sample_a', 'sample_b'], fontsize=14)
    plt.vlines(x=sample_a[col].mean(), ymin=0, ymax=0.2, color='blue', linestyle='--')
    plt.vlines(x=sample_b[col].mean(), ymin=0, ymax=0.2, color='red', linestyle='--')
    plt.title('distribution of residual sugar for the 2 samples')
    plt.savefig('06_dist_residual_sugar_2_samples.png', bbox_inches='tight')
    plt.show()


def run_random_forest_classification():
    df = pd.read_csv("data/winequality-red.csv", sep=';')
    df['type'] = 'red'
    df2 = pd.read_csv("data/winequality-white.csv", sep=';')
    df2['type'] = 'white'
    df = pd.concat([df, df2])
    assert df.isnull().values.sum() == 0, 'nan value in dataset'

    print(df.info())

    enc, df = utils.one_hot_encode_categorical_cols(df, ['type'])

    # dropping column with high correlation
    df = df.drop('total sulfur dioxide', axis=1)
    df['target'] = [1 if x >= 7 else 0 for x in df['quality']]

    X = df.drop(['quality', 'target'], axis=1)
    y = df['target']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    norm = MinMaxScaler()
    norm_fit = norm.fit(x_train)
    x_train_norm = norm_fit.transform(x_train)
    x_test_norm = norm_fit.transform(x_test)

    rf_cls = RandomForestClassifier()
    model = rf_cls.fit(x_train_norm, y_train)

    y_test_pred = rf_cls.predict(x_test_norm)
    accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print(f'accuracy for test data, {accuracy}')

    mse = metrics.mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    print(f'mse, {mse}, rmse, {rmse}')

    report = classification_report(y_test, y_test_pred)
    print(report)

    # plot feature importance
    importances = model.feature_importances_
    cols = df.columns
    """
    for i, v in enumerate(importances):
        print('Feature: %0d, Score: %.5f' % (i, v))
    """
    features_table = utils.get_importance_features_table(importances, cols)
    plt.bar(features_table.keys(), features_table.values())
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.savefig('07_feature_importance.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # sources
    # https://archive.ics.uci.edu/ml/datasets/wine+quality

    run_eda()
    run_random_forest_classification()
    #utils.output_markdown()
