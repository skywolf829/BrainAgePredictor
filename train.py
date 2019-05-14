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
from scipy import stats
import utility_functions
from sklearn.base import clone

patientToGroup, patientToAge, patientToFeatures, headers = utility_functions.loadCSVDictionary('cortical_data.csv')
patientToGroup, patientToAge, patientToFeatures, headers = utility_functions.loadCSVDictionary('subcortical_data.csv', patientToGroup, patientToAge, patientToFeatures, headers)
patientToGroup, patientToAge, patientToFeatures, headers = utility_functions.loadCSVDictionary('DTI_data.csv', patientToGroup, patientToAge, patientToFeatures, headers)

IDs, groups, ages, features = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [0])
IDs1, groups1, ages1, features1 = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [1])
IDs2, groups2, ages2, features2 = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [2])
IDs3, groups3, ages3, features3 = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [3])

IDs_all, groups_all, ages_all, features_all = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [0, 1 , 2, 3])

#features_all = np.concatenate(features_all[0], utility_functions.loadCSV('subcorticalData.csv')[5][0], utility_functions.loadCSV('DTIdata.csv')[5][0])
#ages_all = np.concatenate(ages_all[0], utility_functions.loadCSV('subcorticalData.csv')[10], utility_functions.loadCSV('DTIdata.csv')[10])

# REGRESSORS
#classifier = svm.SVR(kernel='rbf', C=15, gamma="scale")
#classifier = LinearRegression()
#classifier = svm.SVR(kernel='linear', C=.0000001, gamma='scale')
#classifier = svm.SVR(kernel='poly', C=50, gamma='scale', degree=4, coef0=1, epsilon=1)
#classifier = MLPRegressor(hidden_layer_sizes=(25), solver="adam", activation='logistic', tol=0.00001, alpha=0.0001, max_iter=1000000)
#classifier = DecisionTreeRegressor()
#classifier = GridSearchCV(svm.SVR(), params, cv=5)
#classifier = RandomizedSearchCV(svm.SVR(), params, cv = 5, n_iter=1000, n_jobs=-1)
#classifier = linear_model.LinearRegression()
#classifier = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
#classifier = linear_model.Lasso(alpha=0.1, max_iter=1000)
#classifier = linear_model.ElasticNetCV()
#classifier = linear_model.LarsCV()
classifier = linear_model.BayesianRidge()
#classifier = linear_model.SGDRegressor(max_iter=10000)
#classifier = linear_model.Perceptron(max_iter=10000)
#classifier = linear_model.HuberRegressor(max_iter=10000)
#classifier = make_pipeline(PolynomialFeatures(5), Ridge())
#classifier = PolynomialFeatures(2)

# CLASSIFIERS
#classifier = svm.SVC(kernel='rbf', C=10, gamma="scale")
#classifier = DecisionTreeClassifier()
#classifier = GaussianNB()
#classifier = BernoulliNB()
#classifier = KNeighborsClassifier()
#classifier = NearestCentroid()
#classifier = RandomizedSearchCV(svm.SVC(), params, cv=3, n_iter=100, n_jobs=-1)


params = {'C': stats.expon(scale=10), 'gamma': ["scale"], 'coef0':[0,1,10], 'kernel': ['sigmoid']}
kFolds = 5
iterations = 100
random_state = 12883823

sc_X = StandardScaler()
sc_X.fit(features)

features = sc_X.transform(features)
#features1 = sc_X.transform(features1)
#features2 = sc_X.transform(features2)
#features3 = sc_X.transform(features3)
#features_all = sc_X.transform(features_all)
features, ages = shuffle(features, ages)

features = np.array(features)
ages = np.array(ages)
X_CV = np.array(features)
X_validation = np.array(features)
y_CV = np.array(ages)
y_validation = np.array(ages)
#X_CV, X_validation, y_CV, y_validation = train_test_split(features, ages, test_size=0.0, random_state=42)
#y_CV = np.array(y_CV)
#classifier.fit(X_train, y_train)
#print("Score on training data: " + str(classifier.score(X_test, y_test)))

#features, ages = shuffle(features, ages)
#ages = np.array(ages)
#y_pred = cross_val_predict(classifier, features, ages, cv=kFolds)


#for currentVar in range(0,len(features[0]-1)):
i = 0
bestScore = 0
averageScore = 0
rollingAverage = 0
bestClassifier = None
rkf = RepeatedKFold(n_splits=kFolds, n_repeats=iterations, random_state=random_state)
#X_CV = features[:,currentVar]
#X_CV = np.reshape(X_CV, (-1, 1))
for train_index, test_index in rkf.split(X_CV):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_CV[train_index], X_CV[test_index]
    y_train, y_test = y_CV[train_index], y_CV[test_index]
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    #print("Score: " + str(score))
    averageScore = averageScore + score
    rollingAverage = rollingAverage + score
    i = i + 1
    if i % kFolds == 0:
        #print("Average for 5-fold CV split " + str(i / kFolds) + ": " + str(rollingAverage / kFolds))
        rollingAverage = 0

    if score > bestScore:
        bestScore = score
        bestClassifier = clone(classifier) 

averageScore = averageScore / i

print("Average score: " + str(averageScore))
#print("Best score: " + str(bestScore))
bestClassifier.fit(X_validation, y_validation)
validationScore = bestClassifier.score(X_validation, y_validation)
print("Validation score: " + str(validationScore))
y_pred = classifier.predict(X_validation)
print("Average difference: " + str(utility_functions.calcAverageDiff(y_pred, y_validation)))
print("Average absolute difference: " + str(utility_functions.calcAbsAverageDiff(y_pred, y_validation)))

fig, ax = plt.subplots()
ax.scatter(y_validation, y_pred, edgecolors=(0, 0, 0))
ax.plot([ages.min(), ages.max()], [ages.min(), ages.max()], 'k--', lw=2)
ax.set_title('Predicted vs Actual Age')
ax.set_xlabel('Actual Age')
ax.set_ylabel('Predicted Age')
plt.show()

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

#utility_functions.printWithLines(y_predAll - ages_all)