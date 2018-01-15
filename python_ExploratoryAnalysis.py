
#%% SQL LIKE FUNCTIONS
""" 
SQL like functions using pandas
https://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html
"""
# importing a dataset to work with

loan= pd.read_csv('train.csv')

''' select top 5 Gender, Married, Dependents from loan '''
loan[['Gender', 'Married', 'Dependents']].head(5)

''' WHERE 
select top 5 * from loan  where Loan_status = 'Y' and Married = 'No' '''
loan[(loan['Loan_Status'] == 'Y') & (loan['Property_Area'] =='Urban')].head(5)

''' select top 5 Loan_ID from loan where Self_Employed is null'''
loan['Loan_ID'][loan['Self_Employed'].isnull()].head(5) 
# The equivalent to is not null = notnull()

''' GROUP BY
select Gender, count(*) from loan group by 1
'''
loan.groupby('Gender').size()
loan.groupby('Gender').count() #gives the count of not null values per column

df.groupby(df.productcode).mean() # all variables
df['sales'].groupby(df.productcode).mean()  #just one variable

df['sales'].groupby([df.productcode, df.cost]).sum()

''' select Gender, avg(ApplicantIncome), count(*) from loan group by 1 '''
loan.groupby('Gender').agg({'ApplicantIncome': np.mean, 'Gender': np.size})

#if more than one column is needed one can pass a list of variables on groupby:
''' select Gender, Education, avg(ApplicantIncome), count(*) from loan group by 1 '''
loan.groupby(['Gender', 'Education']).agg({'ApplicantIncome': [np.mean, np.size]})

''' JOINS '''
# create a dataset to play with
avgLoan= loan['ApplicantIncome'].groupby(loan.Gender).mean().reset_index(name='avgloan') 
Loan2 = loan.head(50)

''' INNER JOIN
select * from Loan2 a inner join avgLoan b on a.Gender = b.Gender '''
#Merge performs Inner joins by default
pd.merge(Loan2, avgLoan, on='Gender')

''' LEFT JOIN
select * from Loan2 a left join avgLoan b on a.Gender = b.Gender - in this case the result is the same'''
pd.merge(Loan2, avgLoan, on='Gender', how='left') 
# change how='rigt' for a right join and how ='outer' for a full join

''' UNION ALL
select * from AvgLoan UNION ALL select * from avgLoan - dub example just for illutration purposes'''
pd.concat([avgLoan, avgLoan])

#Join by Index
pd.merge(sub1, tmp, left_index = True, right_index= True)

''' UNION 
select * from AvgLoan UNION  select * from avgLoan - dub example just for illutration purposes'''
pd.concat([avgLoan, avgLoan]).drop_duplicates()

''' UPDATE
update avgLoan
set avgloan = avgloan*2
where Gender = 'Male'
'''
avgLoan.loc[avgLoan['Gender']=='Male', 'avgloan'] *=2

'''DELETE
delete from avgLoan where Gender = 'Male
In pandas we select those that should remain instead of deleting them '''
avgLoan = avgLoan.loc[avgLoan['Gender']!= 'Male']
# Drop specific column - either one works
Loan2.drop(['Loan_Status'], axis=1)
del(Loan2['Loan_Status'])

#%% StatisticaL OPERATIONS

Xy2.station_avg_temp_c[Xy2.city=='sj'].mean()
Xy2.station_avg_temp_c[Xy2.city=='sj'].hist(bins=100)

#Outliers 

import numpy as np

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

#  Hypothesis tests
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

# ANOVA
anova=smf.ols(formula='total_cases ~ C(month)',data=Xy2).fit()
anova.summary()
# at least 1 of the means is different according to the anova test
Xy2.groupby(['city']).agg({'total_cases':np.mean})

Xy2.boxplot('total_cases', by='city')

# Perform a post hoc test: Tukey Honest Significance Diff HSD
mc= multi.MultiComparison(Xy2['total_cases'], Xy2['month'])
res= mc.tukeyhsd()
print(res.summary())

# Pearson corr
Xy2.corr() # gives us only corr coefs. pearson r gives pvalues
stats.pearsonr(Xy2.reanalysis_min_air_temp_k, Xy2.total_cases)
stats.pearsonr(Xy2.year, Xy2.total_cases)

sns.regplot(x='year', y='total_cases', data=Xy2)
plt.plot(Xy2.reanalysis_air_temp_k, Xy2.total_cases, 'bo')

# Chisq
import scipy.stats as stats

# create a cat variable based on quantiles
def mean_radius_cat(meanval):
    if meanval < 11.7:
        return 1
    elif (meanval > 11.7) & (meanval< 15.78):
        return 2
    else:
        return 3
    
df['mean_radius_cat']= df.mean_radius.map(mean_radius_cat)
# then create a contingency table 
ct1= pd.crosstab(df.has_cancer, df.mean_radius_cat)

col_sum = ct1.sum(axis=0)
col_pct= ct1/col_sum

#Chisq
cs1 = stats.chi2_contingency(ct1)
print(ct1)
print(col_pct)
print(cs1) #the first 2 values are chi2 and its pvalue

# post hoc tests because expl var has more than 2 levels
# Bonferroni adjustment to avoid familywise error: P/#comparisons
# in this case 0.05/3 = 0.0166 is our new sig level for each comparison
mean1_2= df[df.mean_radius_cat.isin([1,2])] #this filters dataframe
ct2= pd.crosstab(mean1_2.has_cancer, mean1_2.mean_radius_cat)
stats.chi2_contingency(ct2)
# this gives the chi2 and pvalue for this comparison. Assess it based in the
# new adjusted alpha level

#%% REGRESSION
''' REGRESSION ANALYSIS '''
# Linear Regression with Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

reg1= smf.ols('''total_cases ~ C(month) + precipitation_amt_mm+ reanalysis_air_temp_k+
        reanalysis_dew_point_temp_k+
       reanalysis_max_air_temp_k+ reanalysis_min_air_temp_k+
       reanalysis_relative_humidity_percent+ reanalysis_tdtr_k+
       station_avg_temp_c+   station_min_temp_c + C(year) + C(city)''', data= Xy2).fit()
print(reg1.summary())

# To use reference cell coding we add the following after a cat variable:
# C(city, Treatment(reference=1))

# QQ plot of normality of errors
sm.qqplot(reg1.resid, line='r')

# Plot of residuals
stdres= pd.DataFrame(reg1.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')


# additional regression diagnostic plots
fig=plt.figure(figsize=(12, 8))
sm.graphics.plot_regress_exog(reg1,  "precipitation_amt_mm", fig=fig)

# leverage plot
sm.graphics.influence_plot(reg1, size=8)

#%% Stepwise selection to see which features are important

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise_selection(X, y)

print('resulting features:')
print(result)

#%% Logistic Regression with statsmodels
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

#use the cancer dataset created at section 1
lreg = smf.logit(formula= 'has_cancer ~ mean_radius + mean_texture', 
                 data = df).fit()
print(lreg.summary())

#odds ratios: prob of an event occuring in one group vs occuring in another
conf= lreg.conf_int()
conf['OR']= lreg.params
conf.columns=['Lower CI', 'Upper CI', 'Odds Ratio']
print(np.exp(conf))