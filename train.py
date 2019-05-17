# Author: Skylar Wurster
# Description: Evaluating different models on different sets of data. 
#   Sets are either related to brain readings, or bodily measurements, 
#   and a model for a "brain-age" and "body-age" were made using 
#   control groups. Schizophrenic patients' ages were predicted using
#   these models to see if advanced aging was present at a statistically
#   significant margin.

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
import utility_functions
from sklearn.base import clone
import pandas

# Importing the files into dictionaries for BodyAge models
# Dictionaries map patientID -> group, age, and features. This does not
# support multiple readings per patient, but can combine different readings
# across multiple CSVs
#patientToGroup, patientToAge, patientToFeatures, headers = utility_functions.loadCSVDictionary('BodyBloodAge_Body.csv')
#patientToGroup, patientToAge, patientToFeatures, headers = utility_functions.loadCSVDictionary('BodyBloodAge_Blood.csv', patientToGroup, patientToAge, patientToFeatures, headers)
#patientToGroup, patientToAge, patientToFeatures, headers = utility_functions.loadCSVDictionary('BodyBloodAge_NewVars.csv')
patientToGroup, patientToAge, patientToFeatures, headers = utility_functions.loadCSVDictionary('BrainAge_DTI_Ages.csv', multiple_readings=True)


# Importing the files into dictionaries for BrainAge models
#patientToGroup, patientToAge, patientToFeatures, headers = utility_functions.loadCSVDictionary('cortical_data.csv')
#patientToGroup, patientToAge, patientToFeatures, headers = utility_functions.loadCSVDictionary('subcortical_data.csv', patientToGroup, patientToAge, patientToFeatures, headers)
#patientToGroup, patientToAge, patientToFeatures, headers = utility_functions.loadCSVDictionary('DTI_data.csv', patientToGroup, patientToAge, patientToFeatures, headers)

# Extract the lists (correctly ordered) for the IDs, groups, ages, and features
# for whatever data was loaded
# Control Group 
IDs, groups, ages, features = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [0], multiple_readings=True)
# Group 1
IDs1, groups1, ages1, features1 = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [1], multiple_readings=True)
# Group 2
IDs2, groups2, ages2, features2 = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [2], multiple_readings=True)
# Group 3
IDs3, groups3, ages3, features3 = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [3], multiple_readings=True)
# All groups
IDs_all, groups_all, ages_all, features_all = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [0, 1 , 2, 3], multiple_readings=True)



# REGRESSORS
# Simple uncomment the regressor model you'd like to use
classifier = LinearRegression()
# SVM with RBF kernel requires some parameter tuning. gamma='scale' does well, C ranges between 1-100
#classifier = svm.SVR(kernel='rbf', C=10, gamma="scale")
#classifier = svm.SVR(kernel='linear', C=10, gamma='scale')
# This is a classical neural net
#classifier = MLPRegressor(hidden_layer_sizes=(25), solver="adam", activation='logistic', tol=0.00001, alpha=0.0001, max_iter=1000000)
#classifier = DecisionTreeRegressor()
#classifier = GridSearchCV(svm.SVR(), params, cv=5)
#classifier = RandomizedSearchCV(svm.SVR(), params, cv = 5, n_iter=1000, n_jobs=-1)
#classifier = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
#classifier = linear_model.Lasso(alpha=0.1, max_iter=1000)
#classifier = linear_model.ElasticNetCV()
#classifier = linear_model.BayesianRidge()
#classifier = linear_model.SGDRegressor(max_iter=10000)
#classifier = make_pipeline(PolynomialFeatures(2), Ridge())

# Parameters for GridSearchCV or RandomizedSearch CV. Hasn't been used for a while. 
# For hyperparameter tuning.
params = {'C': stats.expon(scale=10), 'gamma': ["scale"], 'coef0':[0,1,10], 'kernel': ['sigmoid']}

# Number of folds in k-fold cross validation. Recommended 5 for adequate training size.
kFolds = 5

# Number of iterations to average the cross validation over. 100 is quick, 1000 might take
# a few seconds but allow for us to make more powerful claims about statistical significance
# of a model being "better"
iterations = 100

# Used in shuffling, allows repeatable results
random_state = 12883823

# Standard scaler simply sets all values to their z-score for their column
sc_X = StandardScaler()
# PCA is priniciple component analysis. Was tested since much of the data
# is multicollinear, but ended up having worse performance.
#sc_X = PCA()

# Fit the scalar to the data that you plan to train on. features is the control group
sc_X.fit(features)
# After fitting the scalar, transform the features to reflect this fit.
features = sc_X.transform(features)
features1 = sc_X.transform(features1)
#features2 = sc_X.transform(features2)
#features3 = sc_X.transform(features3)
features_all = sc_X.transform(features_all)
# Not necessary to shuffle since CV (cross validation) does that for us
#features_all, ages_all = shuffle(features_all, ages_all, random_state=random_state)

# Need to have numpy arrays for next steps
features = np.array(features)
features_all = np.array(features_all)
ages = np.array(ages)
ages_all = np.array(ages_all)

# Set the training data you'd like with X_CV and y_CV
# Set the final test/fitting data (after CV) with X_validation and y_validation
X_CV = features
X_validation = features
y_CV = ages
y_validation = ages

# The following is a visualization for the correlation matrix to
# see if multicollinearity is an issue. Comment these lines out
# if you don't need to see the graph
data = pandas.DataFrame(data=features, columns=headers)
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, cmap=cm.get_cmap("coolwarm"), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(headers),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(headers, rotation=90)
ax.set_yticklabels(headers)
plt.show()

# i keeps track of how many CVs we've done
i = 0
# keeps track of the best score during CV, no longer used
bestScore = 0
# used to keep track of the average score for all CVs
averageScore = 0
# averages over each CV 
rollingAverage = 0
# used to copy the best classifier here, no longer used
bestClassifier = None

# Sets up the repeated kfold, variables are initialized above
rkf = RepeatedKFold(n_splits=kFolds, n_repeats=iterations, random_state=random_state)

# Old code for testing single variable correlation, ignore
#X_CV = features[:,currentVar]
#X_CV = np.reshape(X_CV, (-1, 1))

# The CV evaluation loop
for train_index, test_index in rkf.split(X_CV):
    # This splits the array into lists to train and test on 
    # according to the CV kfold you used
    X_train, X_test = X_CV[train_index], X_CV[test_index]
    y_train, y_test = y_CV[train_index], y_CV[test_index]
    # fit the classifier on the training
    classifier.fit(X_train, y_train)
    # score the classifier on the test set
    score = classifier.score(X_test, y_test)
    # print the score if you'd like
    #print("Score: " + str(score))
    # tally up the averages
    averageScore = averageScore + score
    rollingAverage = rollingAverage + score
    i = i + 1
    # Can optionally print out the CV average after the k-folds have been evalutated
    if i % kFolds == 0:
        #print("Average for 5-fold CV split " + str(i / kFolds) + ": " + str(rollingAverage / kFolds))
        rollingAverage = 0

    # No longer used
    if score > bestScore:
        bestScore = score
        bestClassifier = clone(classifier) 

# Divide by numIterations to get the actual average score
averageScore = averageScore / i

# Print the score
print("Average score: " + str(averageScore))

# Fit the classifier to the groups you'd like
classifier.fit(features, ages)
# Get the score for this fit.
# Means nothing if the cross validation score was low (meaning it can't generalize)
validationScore = classifier.score(features, ages)
print("Control group score: " + str(validationScore))
group1Score = classifier.score(features1, ages1)
print("Group 1 score: " + str(group1Score))
# use classifier.predict to store the predicted ages (in the order of IDs, groups, ages)
y_pred = classifier.predict(features_all)
y_pred_control = classifier.predict(features)
y_pred_group1 = classifier.predict(features1)
# print("Average difference: " + str(utility_functions.calcAverageDiff(y_pred, ages_all)))
#print("Average absolute difference: " + str(utility_functions.calcAbsAverageDiff(y_pred, ages_all)))

print("Average difference in control: " + str(utility_functions.calcAverageDiff(y_pred_control, ages)))
print("Average absolute difference in control: " + str(utility_functions.calcAbsAverageDiff(y_pred_control, ages)))
print("Average difference in group 1: " + str(utility_functions.calcAverageDiff(y_pred_group1, ages1)))
print("Average absolute difference in group 1: " + str(utility_functions.calcAbsAverageDiff(y_pred_group1, ages1)))


# Graphing the final results
fig, ax = plt.subplots()
ax.scatter(ages_all, y_pred, edgecolors=(0, 0, 0))
ax.plot([ages.min(), ages.max()], [ages.min(), ages.max()], 'k--', lw=2)
ax.set_title('Predicted vs Actual Age\nR^2='+str(validationScore))
ax.set_xlabel('Actual Age')
ax.set_ylabel('Predicted Age')
plt.show()


# The rest is old code to look at each groups results. Has not been used for a while.

#scores = cross_val_score(classifier, X_train, y_train, cv=kFolds)
#print("CV scores: " + str(scores))
#print("Average score: " + str(np.array(scores).mean()))
#classifier.fit(X_train, y_train)
#print("Score on group 1: " + str(classifier.score(features1, ages1)))
#print("Score on group 2: " + str(classifier.score(features2, ages2)))
#print("Score on group 3: " + str(classifier.score(features3, ages3)))

#y_predAll = classifier.predict(features_all)
#y_pred1 = classifier.predict(features1)
#y_pred2 = classifier.predict(features2)
#y_pred3 = classifier.predict(features3)

#print("Avg age diff on group 1: " + str(utility_functions.calcAverageDiff(y_pred1, ages1)))
#print("Avg age diff on group 2: " + str(utility_functions.calcAverageDiff(y_pred2, ages2)))
#print("Avg age diff on group 3: " + str(utility_functions.calcAverageDiff(y_pred3, ages3)))


# Used when I am asked for listing the predicted ages. These will print in the
# correct order and can be copy and pasted into excel columns

#utility_functions.printWithLines(IDs_all)
#print('break')
#utility_functions.printWithLines(groups_all)
#print('break')
#utility_functions.printWithLines(ages_all)
#print('break')
utility_functions.printWithLines(y_pred)