# # Hypothesis Testing Examples

# In this notebook I will perform hypothesis testing analysis on some problems that I found online.

import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from IPython.display import display
sns.set_palette("pastel")
plt.rcParams["figure.figsize"] = (20, 9)
np.random.seed(42)

# # Hypothesis Testing with One Sample
# * Examples are from [Introductory Business Statistics](https://openstax.org/books/introductory-business-statistics/pages/9-4-full-hypothesis-test-examples)

# **Example 9.8**

# **Method 1**
# **t statstics**
n = 15
t_stat = (16-16.43) / (0.8/np.sqrt(n))
print(t_stat)

# **critical value for left tailed test**
t_critical = stats.t.ppf(q=0.05, df=n-1)
print(t_critical)

# **p value**
p_value = stats.t.sf(np.abs(t_stat), n-1)
print(p_value)

# **Method 2**
mu = 16
sigma = 0.8
s = np.random.normal(mu, sigma, n)
print(s.mean())
t_stat, p_value = stats.ttest_1samp(s, popmean=16.43, alternative='less')
print(t_stat, p_value)

x = np.arange(-3, 3, 0.001)
plt.plot(x, stats.norm.pdf(x, 0, 1))
z = x[x<t_stat]
plt.fill_between(z, 0, stats.norm.pdf(z, 0, 1))
plt.axvline(0, color='b', linestyle='--', label=0)
plt.axvline(t_critical, color='k', linestyle='--', label=f'{t_critical:0.2f}')
plt.axvline(t_stat, color='r', linestyle='--', label=f'{t_stat:0.2f}')
plt.title(f"z < {t_stat:0.2f}")
plt.legend()
plt.show()

# **Example 9.9**

# **Method 1**
# **test statstics**
t_stat = (108-100) / (12/np.sqrt(16))
print(t_stat)

# **critical value for right tailed test**
t_critical = stats.t.ppf(q=1-0.05, df=(16-1))
print(t_critical)

# **p value**
p_value = stats.t.sf(np.abs(t_stat), 16-1)
print(p_value)

x = np.arange(-3, 3, 0.001)
plt.plot(x, stats.norm.pdf(x, 0, 1))
z = x[x>t_stat]
plt.fill_between(z, 0, stats.norm.pdf(z, 0, 1))
plt.axvline(0, color='b', linestyle='--', label=0)
plt.axvline(t_critical, color='k', linestyle='--', label=f'{t_critical:0.2f}')
plt.axvline(t_stat, color='r', linestyle='--', label=f'{t_stat:0.2f}')
plt.title(f"z > {t_stat:0.2f}")
plt.legend()
plt.show()

# * t_stat is larger than critical value
# * p_value is smaller than 0.05
# * Therefore we should reject null hypothesis that H0: mu <= 100

# **Method 2**
mu = 108
sigma = 12
s = np.random.normal(mu, sigma, 16)
print(s.mean())
t_stat, p_value = stats.ttest_1samp(s, popmean=100, alternative='greater')
print(t_stat, p_value)

# * p_value is smaller than 0.05
# * we should reject null hypothesis

# **Confidence Interval**
alpha = 0.05
dof = 16 - 1
# percent-point function or quantile function of the t-distribution
t = stats.t.ppf(1 - (alpha / 2), dof)
# standard error of mean
sem = np.std(s, ddof=1) / np.sqrt(16)
# margin of error
d = t * sem
upper_ci = s.mean() + d
lower_ci = s.mean() - d
display(f'95% confidence interval is between {lower_ci} to {upper_ci}')


# **Example 9.10**
t_stat = (7.91-8.0) / (np.sqrt(0.03)/np.sqrt(35))
print(t_stat)
t_critical = stats.norm.ppf(q=(1-0.005))
print(t_critical)
p_value = stats.norm.sf(np.abs(t_stat)) * 2
print(p_value)

# * At 99% significance level the critical value is 2.575
# * t-test of 3.07 is larger than 2.575
# * We should reject the null hypothesis that the machine is filling properly at the mean of 8 ounces
# * The machine will need repair

# # Hypothesis Test for Proportions
# **Example 9.11**
# * 1 sample proportion test
# * H0: p = 0.5
# * H1: p != 0.5

n = 100
p = 0.5
q = 0.5
test_stat = (0.53 - 0.5)/np.sqrt((p*q)/n)
print(test_stat)

critical_value = abs(stats.norm.ppf(0.05/2))
print(critical_value)

p_value = stats.norm.sf(np.abs(test_stat)) * 2
print(p_value)

# * Test statistics of 0.6 is a lot less than the critical value 1.96
# * p-value of 0.54 does not indicate strong difference of sample estimate from hypothesis
# * We cannot reject the null hypothesis that 50% of first time loan borrower are the same in terms of loan size from other borrowers

# # Comparing Two Independent Popluation Means
# [Reference](https://openstax.org/books/introductory-business-statistics/pages/10-1-comparing-two-independent-population-means)
