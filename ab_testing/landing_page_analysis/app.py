# # AB Testing On Landing Page Data

# We will try to analyse which landing page version is more popular from the data gathered.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from IPython.display import display
sns.set_palette("pastel")
plt.rcParams["figure.figsize"] = (20, 9)


# ### Load dataset
df = pd.read_csv('data/ab_data.csv')

# ### Display the usual data about this dataset
display(df.head())

display(df.info())

display(df.columns)

display(df.nunique())

display(df.describe())

# #### Check for null values
assert df.isnull().values.sum() == 0, 'error null values'
display('no null value')

# # Clean up data and remove duplcates
display(df.query('group == "treatment"').shape[0])
display(df.query('landing_page == "new_page"').shape[0])

# #### Drop rows that do not have the correct data
false_index = df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == False].index
display(false_index.shape[0])
df2 = df.drop(false_index)

display(df2.query('group == "treatment"').shape[0])
display(df2.query('landing_page == "new_page"').shape[0])
assert df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0] == 0

# #### Total unique users
display(df2['user_id'].nunique())

# #### Check duplicated users
dup_u = df2[df2.duplicated(['user_id'], keep=False)]
print(dup_u)

# #### Drop the first duplicate user
df2.drop(dup_u.index[0], inplace=True)
dup_u = df2[df2.duplicated(['user_id'], keep=False)]
assert len(dup_u) == 0

# #### Overall conversion rate
p_overall = len(df2.query('converted == 1')) / len(df2)
display(p_overall)

# #### Control Group Conversion Rate
p_control = len(df2.query('group == "control" & converted == 1')) / len(df2.query('group == "control"'))
display(p_control)

# #### Treatment Group Conversion Rate
p_treat = len(df2.query('group == "treatment" & converted == 1')) / len(df2.query('group == "treatment"'))
display(p_treat)

# #### The probability of landing on the new page
p_new = len(df2.query('landing_page == "new_page"')) / len(df2)
display(p_new)

# ## A/B Testing

# We need to do A/B testing by comparing two independent population proportions

# #### Hypothesis
# * H0: p_old = p_new
# * H1: p_new - p_old > 0

# At 5% significance level the test statistics would have to larger than critical value of
critical_val = stats.norm.ppf(1-(0.05/2))
print(critical_val)

# #### Method 1 - by calculation
n_old = df2.query('group == "control"').shape[0]
n_old_converted = df2.query('group == "control" & converted == 1').shape[0]
n_new = df2.query('group == "treatment"').shape[0]
n_new_converted = df2.query('group == "treatment" & converted == 1').shape[0]
print(n_old)
print(n_new)
print(n_old_converted)
print(n_new_converted)

p_c = (n_old_converted + n_new_converted) / (n_old + n_new)
print(p_c)

p_old = n_old_converted / n_old
p_new = n_new_converted / n_new

test_stat = (p_new - p_old) / np.sqrt(p_c * (1-p_c) * (1/n_old+1/n_new))
print(test_stat)

p_value = stats.norm.sf(test_stat)
print(p_value)

# ### Method 2 - by statistical package
z_score, p_value = sm.stats.proportions_ztest([n_new_converted, n_old_converted], [n_new, n_old], alternative='larger')
print(z_score, p_value)

# ### Method 3 - by sampling under the null hypothsis

# conversion rate under the null hypothesis
p_null = len(df2.query('converted == 1')) / len(df2)
p_old = p_null
p_new = p_null
p_diffs = list()
for i in range(0, 10000):
    n_old_converted = np.random.choice([1, 0], size=n_new, replace=True, p=(p_old, 1-p_old))
    n_new_converted = np.random.choice([1, 0], size=n_new, replace=True, p=(p_new, 1-p_new))
    p_diffs.append(n_new_converted.mean() - n_old_converted.mean())

observed_diff = p_treat - p_control
p_diffs = np.array(p_diffs)
null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size)
plt.hist(null_vals)
plt.axvline(observed_diff, color='r')
plt.show()
p_value = (p_diffs > observed_diff).mean()
print(p_value)

# * test statistics is lower than critical value
# * p value is larger than 0.05
# * There is not enough evidence to reject the null hypothesis
