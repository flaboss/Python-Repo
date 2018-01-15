''' MACHINE LEARNING WITH SCIKIT LEARN
In a linear model the MSE, mean squared error is an important measure of
accuracy. SUm(Ŷi-Yi)²/n . 
Bias variance tradeoff= more complex models have low variance and high bias.
The opposite is also true.  

    SUPERVISED LEARNING
'''
#%% LINEAR REGRESSION
# import a sample dataset: house prices in Boston
from sklearn.datasets import load_boston
boston = load_boston()

X= boston['data']
y= boston['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=123)

from sklearn.linear_model import LinearRegression
linreg= LinearRegression().fit(X_train, y_train)

print('Coefficients: {}'.format(linreg.coef_))
print('Intercept: {}'.format(linreg.intercept_))

print('R² Training: {}'.format(linreg.score(X_train, y_train)))
print('R² Test: {}'.format(linreg.score(X_test, y_test)))


''' The R² are ok and are close to each other. If train were much higher
than test, then there is aclear sign of overfitting. Ridge regression
adds a penalty factor to the model to avoid overfitting
'''

#%% RIDGE REGRESSION
# As stated above it is a least squares model that adds a penalty factor to avoid overfitting
# the penalty factor is called regularization
# use the same data as above

from sklearn.linear_model import Ridge
ridge= Ridge(alpha=10).fit(X_train, y_train)

ridge.score(X_train,y_train)
ridge.score(X_test, y_test)
''' in this case our previous model was not overfitting, so the R²
is pretty much the same. Alpha is the penalty factor. Increasing
alpha forces coefficients to move more towards zero. 
very small alpha makes the model like a linear regression model.
Ridge regression is useful when we see overfitting and when we have few
datapoints, which usually leads to overfitting '''

# We can feature normalize the data (make transformatios in the variables) if necessary
# Example:

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled= scaler.fit_transform(X_test)

ridge= Ridge(alpha=10).fit(X_train_scaled, y_train)

ridge.score(X_train_scaled,y_train)
ridge.score(X_test_scaled, y_test)

#in this case it decreased the R², so not useful.

# We can try also different alpha values
import numpy as np

print('Ridge regression: effect of alpha regularization parameter\n')
for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
    r2_train = linridge.score(X_train_scaled, y_train)
    r2_test = linridge.score(X_test_scaled, y_test)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, \
          r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(this_alpha, num_coeff_bigger, r2_train, r2_test))

''' as said, smaller alpha levels tend to approach regular linear
regression models '''

#%% LASSO REGRESSION

# Adds a different type of regularization called L1. This can cause some coeff be =0

# use the same data
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

''' As in ridge regression, lower alphas make it closer to a 
normal linear regression and adds more freatures (variables) to the model
Here we can also try the same for loop used above to try out several values
for alpha, but it is not necessary.

Another type of Lasso regression is the LARS: Least absolute selection
shrinkage option. It usually delivers better prediction than linear models
because it also has a penalty factor that reduces irrelevant features to 0.
In a way it works like a Stepwise selection for models. LAR stands for Least
angle Regression. '''

import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import scale as scale
# import a sample dataset: house prices in Boston
from sklearn.datasets import load_boston
boston = load_boston()

X= boston['data']
y= boston['target']

# standardize variables to have mean of 0 and std of 1
Xs = X.copy()
Xs = scale(Xs.astype('float64'))

X_train, X_test, y_train, y_test = train_test_split(Xs,y, test_size = 0.3,
                                                    random_state = 123)

#Lasso regression with cross validation
lasso= LassoLarsCV(cv=10).fit(X_train, y_train)

# R-square from training and test data
rsquared_train=lasso.score(X_train,y_train)
rsquared_test=lasso.score(X_test,y_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(y_train, lasso.predict(X_train))
test_error = mean_squared_error(y_test, lasso.predict(X_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# print variable names and regression coefficients (if 0 not selected)
dict(zip(boston.feature_names, lasso.coef_))

# plot coefficient progression
m_log_alphas = -np.log10(lasso.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, lasso.coef_path_.T)
plt.axvline(-np.log10(lasso.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(lasso.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, lasso.cv_mse_path_, ':')
plt.plot(m_log_alphascv, lasso.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(lasso.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')

#%% POLYNOMIAL REGRESSION
''' a linear model with a polynomial element. It makes a quadratic
function that can better fit to the data, but we also have the tendency to overfit.
'''
# Use the same data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

linreg= LinearRegression().fit(X_train_poly, y_train)

print('Coefficients: {}'.format(linreg.coef_))
print('Intercept: {}'.format(linreg.intercept_))

print('R² Training: {}'.format(linreg.score(X_train_poly, y_train)))
print('R² Test: {}'.format(linreg.score(X_test_poly, y_test)))

''' The R² increased. we need to be careful because there is the risk of
overfitting the data 
'''

#%% 
'''LOGISTIC REGRESSION
apply it in the cancer dataset'''
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()

X= cancer['data']
y= cancer['target']

# Data split and Fit the Logistic Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)

logreg= LogisticRegression(C=100).fit(X_train,y_train)

# Same as R² 
print('Train Accuracy: {}'.format(logreg.score(X_train,y_train)))
print('Test accuracy: {}'.format(logreg.score(X_test,y_test)))

''' Here the C parameter corresponds to regularization (the penalty factor)
Higher values of C corresponds to Less Regularization in the model, i.e. Logistic
regression tries to find the best model without adding penalty factors.
HIGHER C tries to better classify each data point.
'''

# print variable names and regression coefficients (if 0 not selected)
names=cancer.feature_names
coeffs=logreg.coef_
coeffs=coeffs.ravel() #transform a 2d array into a 1d (others are flatten, flat)
dict(zip(names, coeffs))

#print Odds ratios:
print('odds ratios:')
dict(zip(names, np.exp(coeffs)))

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(y_train, logreg.predict(X_train))
test_error = mean_squared_error(y_test, logreg.predict(X_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

#%%
''' SUPPORT VECTOR MACHINES - SVM or LSVM
It is a classification problem where we draw a line between the datapoints
to separate them. It uses a function to classify based on the sign output:
f(x,w,b)=sign(w 'dotprod' x, +b)

There is also the regularization parameter C as in Log Reg. Larger values of C
means less regularization - fit the datapoints as well as possible.
A smaller value of C means that it is mor tolerant to errors - more regularization.
'''
# application in the same Cancer dataset used in Logistic Regression:
from sklearn.svm import LinearSVC

SVM= LinearSVC(C=10).fit(X_train, y_train)

print('Accuracy on Train data: {}'.format(SVM.score(X_train, y_train)))
print('Accuracy on Test data: {}'.format(SVM.score(X_test, y_test)))

#%% 
''' SVM on a multiclass dataset.
The example above was a binary classification. We can also use SVM to classify
multiclass data like the one done previously with KNN.
The approach here is Onve vs the rest. Sklearn takles one group vs the rest
at a time.

Example using the same fruit dataset as used in the KNN example:
'''
import os 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir('/home/flavio/python/Python_DS_COursera/machine_Learning')
fruits = pd.read_table('fruit_data_with_colors.txt')

#divide explanatory and response variables and then split the data
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)
print('Coefficients:\n', clf.coef_)
print('Intercepts:\n', clf.intercept_)

clf.score(X_train, y_train)
clf.score(X_test, y_test)

plt.scatter(X_train['width'], X_train['height'],  c = y_train, marker = 'o')

#%%
''' Notes on Linear models
The main parameter of linear models is the regularization parameter, called alpha in
the regression models and C in LinearSVC and LogisticRegression . Large values for
alpha or small values for C mean simple models. In particular for the regression mod‐
els, tuning these parameters is quite important. Usually C and alpha are searched for
on a logarithmic scale. The other decision you have to make is whether you want to
use L1 regularization or L2 regularization. If you assume that only a few of your fea‐
tures are actually important, you should use L1. Otherwise, you should default to L2.
L1 can also be useful if interpretability of the model is important. As L1 will use only
a few features, it is easier to explain which features are important to the model, and
what the effects of these features are.

Linear models are very fast to train, and also fast to predict. They scale to very large
datasets and work well with sparse data. If your data consists of hundreds of thou‐
sands or millions of samples, you might want to investigate using the solver='sag'
option in LogisticRegression and Ridge , which can be faster than the default on
large datasets.
'''
#%% 
''' Kernelized SVMs
Sometimes the linear SVM classifier will not capture all the dimensions that are in the
data. That is why Kernelized SVM treats the data before modeling. It basically
normalizes the data and adds a new plane to it, like a cube so that it becomes
easier to classify data based on clusters.

'''
# we import a diff module of sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#use the same cancer dataset 
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer.keys()

X= cancer['data']
y= cancer['target']

cancer['feature_names'] #names of data columns

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=123)

#fit a classifier without normalizing the data:
clf= SVC().fit(X_train, y_train)
print('Accuracy Train: {}'.format(clf.score(X_train, y_train)))
print('Accuracy Test: {}'.format(clf.score(X_test,y_test)))
# without training the data there is sign of overfitting

#%% Now, normalize the data with MINMAX scaling feature preprocessing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.fit_transform(X_test)

clf = SVC(C=10).fit(X_train_scaled, y_train)
print('Accuracy Train: {}'.format(clf.score(X_train_scaled, y_train)))
print('Accuracy Test: {}'.format(clf.score(X_test_scaled,y_test)))
# Accuracy increases and there is no longer sign of overfitting

'''
Notes regaring Kernelized SVMs:
- The transformation that the data suffers can be polynomial (like in the 
polynomial regression) or RBF Kernel (Radial Basis Function - based on an
infinite dimensional Gaussian space)
Along with the C parameter we can also provide a parameter called gamma.
A small gamma means a larger radius for the Gaussian Kernel, which means that
many points are considered close by. High values of gamma make a more complex
model because it considers less data points for each kernel (we then have more
kernels). By default C=1 and gamma=1.
It works well with high dimensional or low dimensional data (few and many variables),
but in datasets above 100,000 samples (observations) may become computationally
expensive.
It also require some preprocessing of the data. That is why many people these
days prefer tree-based models such as random forests or gradient boosting, since
it requires less preprocessing.
'''

#%% 
''' A NOTE ON CROSS VALIDATION 
Examples of K folds cross validation to assess the model
It can be used for regression, classification, svm...
'''
# Ex using  Logistic Regression  model
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X= cancer['data']
y= cancer['target']

from sklearn.model_selection import cross_val_score
# here we will not split in train/test bcause cross valid does it
logreg= LogisticRegression(C=100).fit(X,y)

#cross validation
cv_scores = cross_val_score(logreg,X,y, cv=3)

print('Cross validation scores:', cv_scores)
print('Mean cross valid score:', np.mean(cv_scores))

#we can ask for specific metrics. Default is accuracy:
print('Cross val AUC: ', cross_val_score(logreg, X,y,scoring='roc_auc', cv=5))
print('Cross val RECALL: ', cross_val_score(logreg, X,y,scoring='recall', cv=5))

#for a list of these metrics we can check
from sklearn.metrics.scorer import SCORERS
print(sorted(list(SCORERS.keys())))

# based on this concept we can create a function to do it:
def cross_valid_no_norm_data(model, X,y):
    cv_scores = cross_val_score(model,X,y)
    print('Cross validation scores:', cv_scores)
    print('Mean cross valid score:', np.mean(cv_scores))
    
# calling it with Log reg above
cross_valid_no_norm_data(logreg, X,y)

'''
This type of code for cross validation can be used with data that is not 
transformed (normalized) like MinMax or Log. For that sklearn has a different
way to do the cross validation: Pipeline
http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
'''
#%% Validation Curve Example still based on cancer data
# This code based on scikit-learn validation_plot example
#  See:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve

param_range = np.logspace(-3, 3, 4)
train_scores, test_scores = validation_curve(LogisticRegression(), X, y, param_name='C',param_range=param_range, cv=3)

print(train_scores)
print(test_scores)

#%% Validation curve plot
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve')
plt.xlabel('$\gamma$ (param_name)')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
lw = 2

#%%
''' DECISION TREES
They are like a set of If -else questions that classifies data points
they are good to know more about the data, like in exploratory analysis.
It is good for calssification as well
In sklearn we can control prunning and maximum depth (steps) in the decision
tree process (max_depth and max_leafnodes)
It is a good method for visualization and works on different types of features (variables)
(categorical, bynary, continuous). As a downside, even after pruning they can
stil overfit. Enseble of decision trees offer a better performance.

max_depth: controls the number of split points. Most coommon way to reduce
complexity and overfitting.
min_samples_leaf: controls the minimum # of datapoints a leaf needs to have
before further splitting 9also decreases complexity)
max_leaf_nodes: limits the total number of leafs in a tree.

In practice, adjusting only one of these is enough to adjust overfitting.
'''
#Running on the Cancer dataset
from sklearn import tree
from sklearn.model_selection import train_test_split
import os

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X= cancer['data']
y= cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=123)

clf= tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=123).fit(X_train,y_train)

print('Train Accuracy:', clf.score(X_train,y_train))
print('Test accuracy: ', clf.score(X_test,y_test))

# Confusion matrix:
predictions= clf.predict(X_test)
from sklearn.metrics import confusion_matrix as cm
cm(y_test, predictions)

#exporting dec tree file:

tree.export_graphviz(clf,out_file= 'tree.dot')
print('Path to tree: ', os.getcwd())
# one can vizualize it using the .dot file and http://webgraphviz.com/ or


#%% 
'''Model Evaluation:
    For any of them we can use confusion matrices and confusion matrices derived metrics
    Just accuracy is not enough to evaluate a model, although it is a first step.
    Here we take the Logistic Regression example again
'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X= cancer['data']
y= cancer['target']
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)

logreg= LogisticRegression(C=100).fit(X_train,y_train)

# EVALUATION:
from sklearn.metrics import confusion_matrix

y_predicted= logreg.predict(X_test)
confusion= confusion_matrix(y_test, y_predicted)
print(confusion)

from sklearn.metrics import classification_report
classification_report(y_test, y_predicted, target_names=['0','1'])

'''
A widely used framework for Machine learning model selection is to split data into 3:
- create an initial train/test split
- do cross validation on the trainning data for model/parameter selection
- save a 3rd heldout set for final model evaluation.

Sometimes accuracy is not the right metric for evaluating a model of a ML problem. Fraud detection
and tumor detection and predicting customers who will respond to an offer are different
problems.
'''

#%% NAIVE BAYES CLASSIFIER
'''
NAIVE BAYES Classifier
Called Naive because it assumes that there is no relation among the features,
but sometimes they are correlated.
Due to this assumption the claassifier runs much faster. It's accuracy is not
as good as some more complex models, but it can approximate it. It works well on
high dimensional data.
They perform really well on text classification.

Types of Naive Bayes classifiers in Scikit learn:
    GaussianNB , BernoulliNB , and MultinomialNB . GaussianNB can be applied to
any continuous data, while BernoulliNB assumes binary data and MultinomialNB
assumes count data (that is, that each feature represents an integer count of some‐
thing, like how often a word appears in a sentence). BernoulliNB and MultinomialNB
are mostly used in text data classification.

Example of Gaussian NB on Cancer dataset:
'''
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X= cancer['data']
y= cancer['target']
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)

# Gaussian Naive Bayes Classifier
nbclf = GaussianNB().fit(X_train, y_train)

print('Train accuracy: ', nbclf.score(X_train, y_train))
print('Test accuracy: ', nbclf.score(X_test, y_test))

#%%
''' Random Forests:
They are an ensamble. Ensambles are groupings of models that, together perform
better than alone.
Random forests are groups of decision trees. They can be used as classifiers or regressors
It does not require feature normalization in most of times. It handles, like
decision trees, many different feature types.
On a down side it is usually really difficult for humans to interpret why
a decision was made.
Like decision trees, it may not be a good choice for high dimensional tasks
(like text classifiers) as compared to simpler linear models. They are the most used
ML models.

Parameters for a random forest:
- n_estimators: number of trees to use. Default is 10. For higher datasets this
number must be increased to reduce overfitting, but increases computations.
- max_features: has a strong effect on performance
- max_depth: controls the number of split points. Most coommon way to reduce
complexity and overfitting. 
- n_jobs: how many paralel processing to use during training (-1 uses all cores in the 
system)
- random_state: the seed for reproduction
- max_features: default works well, has strong effect on performance. Usually 
small values reduce overfitting.Rule of thumb max_features=sqrt(n_features) for classification and max_fea
tures=log2(n_features) for regression.

Below an example of random forest classifier on the Cancer data set.
'''

from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X= cancer['data']
y= cancer['target']
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)

clf = RandomForestClassifier(max_features = 8, random_state = 0)
clf.fit(X_train, y_train)

print('Train accuracy: ',clf.score(X_train, y_train))
print('Test accuracy: ', clf.score(X_test, y_test))

# For classification purposes I can request a confusion matrix as before

''' we can plot the most important features in a random forest model
or in a decision tree model using the function below
'''
import matplotlib.pyplot as plt
import numpy as np

# where model is a decision tree or random forest model

def plot_feature_importances_cancer(model, train_data, feature_names):
    n_features = train_data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
feature_names = cancer.feature_names
plot_feature_importances_cancer(clf, X_train, feature_names)


"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction - n_estimators
"""
trees=range(25)
accuracy=np.zeros(25)

for x in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators= x + 1)
   classifier=classifier.fit(X_train,y_train)
   predictions=classifier.predict(X_test)
   accuracy[x]=classifier.score(X_test, y_test)
   
plt.cla()
plt.plot(trees, accuracy)

#%%
''' GRADIENT BOOST DECISION TREES GBDT
One of the best supervised machine learning algorithms nowadays. 

The main parameters are:
- n_estimators: number of decision trees to be used in the 
ensamble (calles weak learners). Usually this is the first adjust to see how heavy
it is on processing.
- learning_rate: controls the emphasis on fixing errors from previous iteration
- max_depth: controls the number of split points. Most coommon way to reduce
complexity and overfitting. Set to small numbers (3-5 in most applications).

The can be used for regression and calssification. The trees are built in a serial 
manner where trees try to correct the errors of previous ones (as opposed to
random forests). 

Again, another example in the cancer dataset
'''
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X= cancer['data']
y= cancer['target']
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)

clf = GradientBoostingClassifier(random_state = 0)
clf.fit(X_train, y_train)

print('Train accuracy: ', clf.score(X_train, y_train))
print('Test accuracy: ', clf.score(X_test, y_test))

''' there is a clear sign of overfitting. We can follow 2 strategies:
    1. apply pre-prunning limiting the depth or
    2. lower the learning rate
'''
# apply pre-pruning:
clf = GradientBoostingClassifier(random_state = 0, max_depth=1)
clf.fit(X_train, y_train)
print('Train accuracy: ', clf.score(X_train, y_train))
print('Test accuracy: ', clf.score(X_test, y_test))

# Lower the learning rate:
clf = GradientBoostingClassifier(random_state = 0, learning_rate=0.01)
clf.fit(X_train, y_train)
print('Train accuracy: ', clf.score(X_train, y_train))
print('Test accuracy: ', clf.score(X_test, y_test))

''' Either of them make the model less overfitting. 
we can also call the function created before to learn which feature plays a
more important role in the model
'''
feature_names = cancer.feature_names
plot_feature_importances_cancer(clf, X_train, feature_names)
#many features got ignored as compared to random forests
# xgboost is a way to implement in large datasets tha works faster than sklearn.
''' In contrast to random forests, where a higher n_esti
mators value is always better, increasing n_estimators in gradient boosting leads to a
more complex model, which may lead to overfitting. A common practice is to fit
n_estimators depending on the time and memory budget, and then search over dif‐
ferent learning_rates.
Another important parameter is max_depth (or alternatively max_leaf_nodes ), to
reduce the complexity of each tree. Usually max_depth is set very low for gradient
boosted models, often not deeper than five splits.
'''

#%% NEURAL NETWORKS
'''
Neural Nets are computational expensive and good to apply when the 
features are of the same kind. They are the basis of Deep Learning.
below are examples of simpler methods (multilayer perceptrons MLPs). 
MLPs can be viewed as generalizations of linear models that perform multiple stages
of processing to come to a decision.

Similar to SVMs they work best with homogenous datasets where all features have similar
meanings and require some preprocessing. Data with different types of features
tree based models is a better solution. Also tunning NN is an art in itself.


Some parameters are:
    - hidden_layer_sizes: sets the number of hidden layers. Default is 100
    - alpha: controls the weight on regularization penalty. default=0.0001
    it constrains the complexity of the model.
    - activation: controls the nonlinear function used for the activation function
    including 'relu' (default), 'logistic', tanh'.
    -max_iter: maximum number of iterations. This helps only in the training data
    not in generalization. 
'''
# Example in the cancer dataset
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler #need to scale the data

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X= cancer['data']
y= cancer['target']
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0,
                    random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)

print('Train accuracy: ', clf.score(X_train_scaled, y_train))
print('Test accuracy: ', clf.score(X_test_scaled, y_test))


''' NN models are hard to understand... this plot shows the weights that
were learned connecting the input to the first hidden layer. The rows in this plot cor‐
respond to the 30 input features, while the columns correspond to the 100 hidden
units. Light colors represent large positive values, while dark colors represent nega‐
tive values:
'''
plt.figure(figsize=(15, 5))
plt.imshow(clf.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()

'''
One possible inference we can make is that features that have very small weights for
all of the hidden units are “less important” to the model. We can see that “mean
smoothness” and “mean compactness,” in addition to the features found between
“smoothness error” and “fractal dimension error,” have relatively low weights com‐
pared to other features. This could mean that these are less important features or pos‐
sibly that we didn’t represent them in a way that the neural network could use.
'''
#%%
''' UNSUPERVISED LEARNING'''
#%% 
'''
 Principal Component Analysis (PCA) and Clustering
'''
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_cancer= cancer['data']
y_cancer= cancer['target']

# Before applying PCA, each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  
# Fit to 2 principal components
pca = PCA(n_components = 2).fit(X_normalized)
X_pca = pca.transform(X_normalized)
print(X_cancer.shape, X_pca.shape)

import matplotlib.pyplot as plt
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_cancer)
#we can clearly see that ther are two clusters. The 30 features were reduced to 2.
# this technique is good to reduce high dimensional datasets.

#%%  Another method of reducing dimensions to a lower dimension space is MDS
# Multidimensional Scaling. Good for visualization and it is similar to PCA

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_cancer= cancer['data']
y_cancer= cancer['target']
# each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  

mds = MDS(n_components = 2)

X_mds = mds.fit_transform(X_normalized)

import matplotlib.pyplot as plt
plt.scatter(X_mds[:,0], X_mds[:,1], c=y_cancer)

#pretty similar to PCA is TSNE, but it is a little trickier to interpret.

#%% K MEANS CLUSTERING
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#very used to create profiles and data reduction techniques
# When dealing with real data it is a good idea to normalize it with
# sklearn.preprocessing.scale or MinMaxScaler

X, y = make_blobs(random_state = 10)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_)

# Example in the fruits dataset
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

os.chdir('/home/flavio/python/Python_DS_COursera/machine_Learning')
fruits = pd.read_table('fruit_data_with_colors.txt')
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']

#data need to be normalized before applying Kmeans
from sklearn.preprocessing import MinMaxScaler
X_fruits_normalized = MinMaxScaler().fit(X).transform(X)  
kmeans = KMeans(n_clusters = 4, random_state = 0)
kmeans.fit(X_fruits_normalized)

# to plot it we need to reduce the dimensions using PCA or MDS.

"""
Good way to determine the number of clusters: Elbow curve plot
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""
# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k).fit(X)
    clusassign=model.predict(X)
    meandist.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) 
    / X.shape[0])
    
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

# Interpret 3 cluster solution
model3=KMeans(n_clusters=3).fit(X)
clusassign=model3.predict(X)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2) #requesting 2 canonical variables for 2d plot
plot_columns = pca_2.fit_transform(X)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

""" ANALYSIS of the clusters
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
X.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(X['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=pd.DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(X, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)

# at this point we can run an ANOVA model to see if the mean values
# of each cluster differ from one another.

"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# NOTE ON KMEANS: it is not too good if the dataset has too many cat variables

#%%   AGROMERATIVE CLUSTER
''' Another Clustering solution is called aglomerative clustering. It builds
upon clusters, clustering them further.
This technique is used when data is too scattered and there is no visual boundary
in the data points.

There are 3 types of Linkage Criteria (to link the clusters):
    1. Wards Method: merge the two clusters that give the least increase in
    total variance among all clusters.
    2. Average Linkage: groups the clusters that have the smallest average
    distance between its points.
    3. Groups the clusters that have the smallest maximum distance of their 
    points.
In general Ward's method works well in most datasets and it is usually the 
method of choice.

brief example with synthetic data:
'''
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

X, y = make_blobs(random_state = 10)

cls = AgglomerativeClustering(n_clusters = 3)
cls_assignment = cls.fit_predict(X)

#%% Parameter Tunning with Grid Search
# Example of a GradientBoost Tree Regressor

from sklearn.model_selection import GridSearchCV
param_test = {'max_depth':range(5, 10), 'n_estimators': range(100, 500, 100)}
gsearch= GridSearchCV(estimator=GradientBoostingRegressor(), 
                      param_grid= param_test,  n_jobs=4, cv=5)

# even if it performs cross validation, we want to use a holdout to test
gsearch.fit(X_train,y_train)

gsearch.grid_scores_
gsearch.best_params_
gsearch.best_score_

final= GradientBoostingRegressor(max_depth=5, n_estimators=100).fit(X_test, y_test)
final.score(X_test, y_test)
