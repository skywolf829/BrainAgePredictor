import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.utils import shuffle
from scipy import stats
import utility_functions

patientToGroup, patientToAge, patientToFeatures = utility_functions.loadCSVDictionary('cortical_data.csv')
patientToGroup, patientToAge, patientToFeatures = utility_functions.loadCSVDictionary('subcortical_data.csv', patientToGroup, patientToAge, patientToFeatures)
patientToGroup, patientToAge, patientToFeatures = utility_functions.loadCSVDictionary('DTI_data.csv', patientToGroup, patientToAge, patientToFeatures)

IDs, groups, ages, features = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [0])
IDs1, groups1, ages1, features1 = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [1])
IDs2, groups2, ages2, features2 = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [2])
IDs3, groups3, ages3, features3 = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [3])

IDs_all, groups_all, ages_all, features_all = utility_functions.getAgesAndFeaturesForGroups(patientToGroup, patientToAge, patientToFeatures, [0, 1 , 2, 3])

#features_all = np.concatenate(features_all[0], utility_functions.loadCSV('subcorticalData.csv')[5][0], utility_functions.loadCSV('DTIdata.csv')[5][0])
#ages_all = np.concatenate(ages_all[0], utility_functions.loadCSV('subcorticalData.csv')[10], utility_functions.loadCSV('DTIdata.csv')[10])

params = {'C': stats.expon(scale=10), 'gamma': ["scale"], 'coef0':[0,1,10], 'kernel': ['sigmoid']}

# REGRESSORS
#classifier = svm.SVR(kernel='rbf', C=50, gamma="scale")
#classifier = LinearRegression()
#classifier = svm.SVR(kernel='linear', C=.0000001, gamma='scale')
#classifier = svm.SVR(kernel='poly', C=0.1, gamma='scale', degree=10, coef0=1, epsilon=1)
#classifier = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver="adam", activation='logistic', tol=0.00001, alpha=0.0001, max_iter=1000000)
#classifier = DecisionTreeRegressor()
#classifier = GridSearchCV(svm.SVR(), params, cv=5)
#classifier = RandomizedSearchCV(svm.SVR(), params, cv = 5, n_iter=1000, n_jobs=-1)
#classifier = linear_model.LinearRegression()
#classifier = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
#classifier = linear_model.Lasso(alpha=0.1, max_iter=1000)
#classifier = linear_model.ElasticNetCV()
#classifier = linear_model.LarsCV()
#classifier = linear_model.BayesianRidge()
#classifier = linear_model.SGDRegressor(max_iter=10000)
classifier = linear_model.Perceptron(max_iter=10000)
#classifier = linear_model.HuberRegressor(max_iter=10000)

# CLASSIFIERS
#classifier = svm.SVC(kernel='rbf', C=10, gamma="scale")
#classifier = DecisionTreeClassifier()
#classifier = GaussianNB()
#classifier = BernoulliNB()
#classifier = KNeighborsClassifier()
#classifier = NearestCentroid()
#classifier = RandomizedSearchCV(svm.SVC(), params, cv=3, n_iter=100, n_jobs=-1)

sc_X = StandardScaler()
sc_X.fit(features)

features = sc_X.transform(features)
features1 = sc_X.transform(features1)
features2 = sc_X.transform(features2)
features3 = sc_X.transform(features3)
features_all = sc_X.transform(features_all)

classifier.fit(features, ages)
print("Score on training data: " + str(classifier.score(features, ages)))
print("Score on group 1: " + str(classifier.score(features1, ages1)))
print("Score on group 2: " + str(classifier.score(features2, ages2)))
print("Score on group 3: " + str(classifier.score(features3, ages3)))

y_predAll = classifier.predict(features_all)
y_pred1 = classifier.predict(features1)
y_pred2 = classifier.predict(features2)
y_pred3 = classifier.predict(features3)

print("Avg age diff on group 1: " + str(utility_functions.calcAverageDiff(y_pred1, ages1)))
print("Avg age diff on group 2: " + str(utility_functions.calcAverageDiff(y_pred2, ages2)))
print("Avg age diff on group 3: " + str(utility_functions.calcAverageDiff(y_pred3, ages3)))

utility_functions.printWithLines(y_predAll - ages_all)