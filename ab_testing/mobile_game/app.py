# # Mobile Game A/B Testing

# I will run do an A/B testing analysis on the cookie cats dataset.  The dataset can be found on Kaggle.
# Basically we would like to find out if there is a difference in user behavior when we place a gate at level 30 and at level 40
# The data contains a control group at gate 30 and a test group at gate 40.
# The user's behavior can be determined by number of games place, and also 1Day/7Day retention status

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
from scipy.stats import shapiro
import scipy.stats as stats
sns.set_palette("pastel")
from IPython.display import display


# ### Load dataset
df = pd.read_csv('data/cookie_cats.csv')

# ### Display the usual data about this dataset
display(df.head())

display(df.info())

display(df.columns)

display(df.nunique())

display(df.describe())

# **we have a outlier value of 49854**

# ### Check for null values
assert df.isnull().values.sum() == 0, 'error null values'
display('no null value')

# ### The range of values in different percentiles
display(df.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["sum_gamerounds"]])

# **50% of the people played less than 16 games**

# Clip data so that outlier points become boundary values
# df['sum_gamerounds'] = df['sum_gamerounds'].clip(lower=df['sum_gamerounds'].quantile(0.01), upper=df['sum_gamerounds'].quantile(0.99), axis=0)
# display(df.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["sum_gamerounds"]])

# #### Remove the outlier in data
# 49854 games played in the first 7 days is unlikely a normal behavior of a normal user
df = df[df['sum_gamerounds'] < df['sum_gamerounds'].max()]

# ### Summary stats about the A/B groups
summary_stats = df.groupby("version")['sum_gamerounds'].agg(["count", "median", "mean", "std", "max"])
display(summary_stats)

# ### Distribution of game rounds values for the 2 groups

# we plot values within the 99% percentile
sns.boxplot(x=df['version'], y=df[df['sum_gamerounds'] <= 500]['sum_gamerounds'])
plt.show()

# ### Histograms for gate 30 and gate 40
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
df.query('sum_gamerounds <= 500')['sum_gamerounds'].plot.hist(title='All', histtype='bar', bins=50, ec='black', ax=axes[0])
df.query('version == "gate_30" & sum_gamerounds <= 500')['sum_gamerounds'].plot.hist(title='Gate 30', histtype='bar', bins=50, ec='black', ax=axes[1])
df.query('version == "gate_40" & sum_gamerounds <= 500')['sum_gamerounds'].plot.hist(title='Gate 40', histtype='bar', bins=50, ec='black', ax=axes[2])

# ### 1D retention for gate 30 and gate 40
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
df.query('retention_1 == True & sum_gamerounds <= 500')['sum_gamerounds'].plot.hist(title='1D All', histtype='bar', bins=50, ec='black', ax=axes[0])
df.query('retention_1 == True & version == "gate_30" & sum_gamerounds <= 500')['sum_gamerounds'].plot.hist(title='1D Gate 30', histtype='bar', bins=50, ec='black', ax=axes[1])
df.query('retention_1 == True & version == "gate_40" & sum_gamerounds <= 500')['sum_gamerounds'].plot.hist(title='1D Gate 40', histtype='bar', bins=50, ec='black', ax=axes[2])

# ### 7D retention for gate 30 and gate 40
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
df.query('retention_7 == True & sum_gamerounds <= 500')['sum_gamerounds'].plot.hist(title='7D All', histtype='bar', bins=50, ec='black', ax=axes[0])
df.query('retention_7 == True & version == "gate_30" & sum_gamerounds <= 500')['sum_gamerounds'].plot.hist(title='7D Gate 30', histtype='bar', bins=50, ec='black', ax=axes[1])
df.query('retention_7 == True & version == "gate_40" & sum_gamerounds <= 500')['sum_gamerounds'].plot.hist(title='7D Gate 40', histtype='bar', bins=50, ec='black', ax=axes[2])

# ### Plot User / Gamerounds value chart for the 2 groups
df[df['version']=='gate_30'].reset_index().set_index('index')['sum_gamerounds'].plot(legend=True, label='Gate 30', figsize=(20, 5))
df[df['version']=='gate_40'].reset_index().set_index('index')['sum_gamerounds'].plot(legend=True, label='Gate 40', alpha=0.8)
plt.suptitle("User/Game rounds", fontsize=20)
plt.show()

# ### Transformation of retention data

# retention_1 and retention_7 columns contain True or False values, we can transform the data into a value counts table
df_retention = pd.DataFrame({'ret_1_count': df['retention_1'].value_counts(),
                             'ret_7_count': df['retention_7'].value_counts(),
                             'ret_1_ratio': df['retention_1'].value_counts() / len(df),
                             'ret_7_ratio': df['retention_7'].value_counts() / len(df),
                             })
display(df_retention)

# **1D retention rate is about 44%, 7D retention rate is about 18%**

# ### Group the games played by user, count the rank the result
games_user_count = df.groupby("sum_gamerounds").userid.count().to_frame()
display(games_user_count.head(20))
games_user_count.loc[0:50]['userid'].plot(legend=True, label='users count', figsize=(20, 9))
plt.show()

# ### Number of people who did not open and play any game
no_play = df.query('sum_gamerounds == 0')
display(no_play.head())
display(len(no_play))

# ### Number of people who have played 30 to 40 games
gated = df.groupby("sum_gamerounds").userid.count().loc[[30, 40]]
display(gated)

# ### Summary statistics for the control and test group
summary_stats = df.groupby("version")['sum_gamerounds'].agg(["count", "median", "mean", "std", "max"])
display(summary_stats)


# # Compare by retention days

# ## Compare 1D retention data for the 2 groups
ret_1d_stat = df.groupby(["version", "retention_1"])['sum_gamerounds'].agg(["count", "median", "mean", "std", "max"])
display(ret_1d_stat)

ret_1d = df.groupby("version")['retention_1'].mean()
display(ret_1d)

# **the 2 retention rate are very close**


# ## Compare at 7D retention data for the 2 groups
ret_7d_stat = df.groupby(["version", "retention_7"])['sum_gamerounds'].agg(["count", "median", "mean", "std", "max"])
display(ret_7d_stat)

ret_7d = df.groupby("version")['retention_7'].mean()
display(ret_7d)

# ### Combine 1D and 7D retention data we can create a new categorical variable
df["RetentionStatus"] = list(map(lambda x, y: str(x)+"-"+str(y), df['retention_1'], df['retention_7']))
retention_status = df.groupby(["version", "RetentionStatus"])['sum_gamerounds'].agg(["count", "median", "mean", "std", "max"])
display(retention_status)

# **we can see that for each grouping the summary stats are similar for gate 30 and 40**
# **next we need to do some tests to see if they are significantly not similar**


# # Test for normality
df = pd.read_csv('data/cookie_cats.csv')
df = df[df['sum_gamerounds'] < df['sum_gamerounds'].max()]

group_a = df[df['version'] == 'gate_30']['sum_gamerounds']
group_b = df[df['version'] == 'gate_40']['sum_gamerounds']
display(group_a)
display(group_b)

shapiro_a = shapiro(group_a)
shapiro_b = shapiro(group_b)
if shapiro_a[1] < 0.05:
    display('group a is not normal')
else:
    display('group a is normal')
if shapiro_b[1] < 0.05:
    display('group b is not normal')
else:
    display('group b is normal')

# **the 2 groups are not normal**

# ### Run Mann-Whitney U Test
utest = stats.mannwhitneyu(group_a, group_b)
display(utest)

# # Test by bootstrapping

# ## Bootstrap 1D retention

# ### Creating an list with bootstrapped means for each gate group
sample_means = []
for i in range(1000):
    boot_mean = df.sample(frac=1, replace=True).groupby('version')['retention_1'].mean()
    sample_means.append(boot_mean)

# ### plot kernel density of the bootstrap distributions
sample_means = pd.DataFrame(sample_means)
sample_means.plot(kind='density')
plt.show()

# ### calculate difference of the means of gate 30 and gate 40
sample_means['diff'] = (sample_means['gate_30'] - sample_means['gate_40']) / sample_means['gate_40'] * 100
ax = sample_means['diff'].plot(kind='density')
ax.set_title('Density of 1D retention mean difference of 2 groups')
plt.show()
ratio = (sample_means['diff'] > 0).mean() * 100
display(f'For 1D retention, gate 30 mean is larger around {ratio}% of the time')

# ## Bootstrap 7D retention

# ### Creating an list with bootstrapped means for each gate group
sample_means = []
for i in range(1000):
    boot_mean = df.sample(frac=1, replace=True).groupby('version')['retention_7'].mean()
    sample_means.append(boot_mean)

# ### plot kernel density of the bootstrap distributions
sample_means = pd.DataFrame(sample_means)
sample_means.plot(kind='density')
plt.show()

# ### calculate difference of the means of gate 30 and gate 40
sample_means['diff'] = (sample_means['gate_30'] - sample_means['gate_40']) / sample_means['gate_40'] * 100
ax = sample_means['diff'].plot(kind='density')
ax.set_title('Density of 7D retention mean difference of 2 groups')
plt.show()
ratio = (sample_means['diff'] > 0).mean() * 100
display(f'For 7D retention, gate 30 mean is larger around {ratio}% of the time')


# # Conclusion

# **From the Mann-Whitney U test result, the p-value is around 0.05**
# **we can say the distributions for the 2 groups are different, but it is hard to say how different these groups are**

# **We then perform the bootstrap method to sample the data**
# **From the bootstrap test result above, in terms of retention rate**
# **placing the gate at level 30 will yield higher 1D retention and 7D retention and is therefore the better strategy**
