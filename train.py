import numpy as np
from sklearn import svm
import matplotlib.pyplot as ptl
import csv
import time
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

def getCols(data, col):
    newData = []
    currentRow = 0
    for row in data:
        currentCol = 0
        newRow = []
        for item in row:
            if currentCol in col:
                newRow.append(item)
            currentCol = currentCol + 1
        newData.append(newRow)
        currentRow = currentRow + 1
    return newData

def dropCols(data, dropCols):
    newData = []
    currentRow = 0
    for row in data:
        currentCol = 0
        newRow = []
        for item in row:
            if not currentCol in dropCols:
                newRow.append(item)
            currentCol = currentCol + 1
        newData.append(newRow)
        currentRow = currentRow + 1
    return newData

def calcAverageDiff(arr1, arr2):
    diff = 0
    i = 0
    while i < len(arr1):
        diff = diff + abs(arr1[i] - arr2[i])
        i = i + 1
    diff = diff / len(arr1)
    return diff

def printWithLines(arr):
    for item in arr:
        print(item)

grouping = 10
features = []
features_1 = []
features_2 = []
features_3 = []
features_1to3 = []
features_all = []
age_groups = []
age_groups_1 = []
age_groups_2 = []
age_groups_3 = []
ages = []
ages_1 = []
ages_2 = []
ages_3 = []
ages_1to3 = []
ages_all = []
headers = []
with open('training_use_0_dataset.csv') as file:
    csv = csv.reader(file)
    r = 0
    for row in csv:
        control = 0
        c = 0
        patientData = []        
        for item in row:
            if r == 0:
                if item != None and item != "" and item != " ":
                    headers.append(item)
            else:
                if item != None and item != "" and item != " ":
                    if c == 1:
                        if item == "0":
                            control = 0
                        if item == "1":
                            control = 1
                        if item == "2":
                            control = 2
                        if item == "3":
                            control = 3
                    elif c == 2:
                        age_bot = int(float(item) / grouping)
                        if control == 0:
                            age_groups.append(str(age_bot * grouping) + " to " + str(age_bot * grouping + grouping - 1)) 
                            ages.append(float(item))
                        elif control == 1:
                            age_groups_1.append(str(age_bot * grouping) + " to " + str(age_bot * grouping + grouping - 1)) 
                            ages_1.append(float(item)) 
                        elif control == 2:
                            age_groups_2.append(str(age_bot * grouping) + " to " + str(age_bot * grouping + grouping - 1)) 
                            ages_2.append(float(item)) 
                        elif control == 3:
                            age_groups_3.append(str(age_bot * grouping) + " to " + str(age_bot * grouping + grouping - 1)) 
                            ages_3.append(float(item)) 
                    if c >= 4 and c <= 28:
                        patientData.append(float(item))
                        
                c = c + 1
        if len(patientData) > 0 and r != 0:
            if control == 0:
                features.append(np.array(patientData))
            elif control == 1:
                features_1.append(np.array(patientData))
            elif control == 2:
                features_2.append(np.array(patientData))
            elif control == 3:
                features_3.append(np.array(patientData))
        r = r + 1
features_1to3 = np.concatenate((features_1, features_2, features_3))
features_all = np.concatenate((features, features_1, features_2, features_3))
ages_1to3 = np.concatenate((ages_1, ages_2, ages_3))
ages_all = np.concatenate((ages, ages_1, ages_2, ages_3))

params = {'C': stats.expon(scale=10), 'gamma': stats.expon(scale=0.1), 'kernel': ['rbf']}#, 'gamma': stats.expon(scale=.1), 'kernel': ['rbf']}
 


# REGRESSORS
classifier = svm.SVR(kernel='rbf', C=10000, gamma=0.1)
#classifier = LinearRegression()
#classifier = svm.SVR(kernel='linear', C=.0000001, gamma='scale')
#classifier = svm.SVR(kernel='poly', C=0.1, gamma='scale', degree=10, coef0=1, epsilon=1)
#classifier = MLPRegressor(hidden_layer_sizes=(100), solver="adam", activation='logistic', tol=0.00001, alpha=0.0001, max_iter=1000000)
#classifier = DecisionTreeRegressor()
#classifier = GridSearchCV(svm.SVR(), params, cv=5)
#classifier = RandomizedSearchCV(svm.SVR(), params, cv = 5, n_iter=100, n_jobs=-1)
#classifier = linear_model.LinearRegression()
#classifier = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
#classifier = linear_model.Lasso(alpha=0.1, max_iter=1000)
#classifier = linear_model.ElasticNetCV()
#classifier = linear_model.LarsCV()
#classifier = linear_model.BayesianRidge()
#classifier = linear_model.SGDRegressor(max_iter=10000)
#classifier = linear_model.Perceptron(max_iter=10000)
#classifier = linear_model.HuberRegressor(max_iter=10000)

# CLASSIFIERS
#classifier = svm.SVC(kernel='rbf', C=10, gamma="scale")
#classifier = DecisionTreeClassifier()
#classifier = GaussianNB()
#classifier = BernoulliNB()
#classifier = KNeighborsClassifier()
#classifier = NearestCentroid()
#classifier = RandomizedSearchCV(svm.SVC(), params, cv=3, n_iter=100, n_jobs=-1)

i = 0
iters = 1

average_score_1 = 0
average_score_2 = 0
average_score_3 = 0
average_age_diff_1 = 0
average_age_diff_2 = 0
average_age_diff_3 = 0

sc_X = StandardScaler()
sc_y = StandardScaler()
pca = PCA(n_components=None)


#ages = sc_y.fit_transform(np.array(ages)[:, np.newaxis]).reshape(len(ages))
#features = pca.fit_transform(features)
#features = sc_X.fit_transform(features)
#print(np.mean(features, axis=0))
#print(np.min(features, axis=0))
#print(np.max(features, axis=0))
#print(np.std(features, axis=0))
#np.histogram(np.array(features).take(1))


while i < iters:
    #features, ages = shuffle(features, ages, random_state=0)
    #X_train, X_test, y_train, y_test = train_test_split(features, ages, test_size = 0.20) 
    
    classifier.fit(features, ages)
    print(classifier.score(features, ages))
    y_predAll = classifier.predict(features_all)
    score1 = classifier.score(features_1, ages_1)
    score2 = classifier.score(features_2, ages_2)
    score3 = classifier.score(features_3, ages_3)
    y_pred1 = classifier.predict(features_1)
    y_pred2 = classifier.predict(features_2)
    y_pred3 = classifier.predict(features_3)
    average_age_diff_1 = average_age_diff_1 + calcAverageDiff(y_pred1, ages_1)
    average_age_diff_2 = average_age_diff_2 + calcAverageDiff(y_pred2, ages_2)
    average_age_diff_3 = average_age_diff_3 + calcAverageDiff(y_pred3, ages_3)

    average_score_1 = average_score_1 + score1
    average_score_2 = average_score_2 + score2
    average_score_3 = average_score_3 + score3
    #average_age_diff = average_age_diff + age_diff
    #print(classifier.best_params_)
    #("Finished step " + str(i+1) + "/" + str(iters) + " score1= " + str(score1) + " score2= " + str(score2) + " score3= " + str(score3)) 
    #print("Age diff= " + str(age_diff))
    i = i + 1


average_score_1 = average_score_1 / iters
average_score_2 = average_score_2 / iters
average_score_3 = average_score_3 / iters
average_age_diff_1 = average_age_diff_1 / iters
average_age_diff_2 = average_age_diff_2 / iters
average_age_diff_3 = average_age_diff_3 / iters
#average_age_diff = average_age_diff / iters
#print("Complete. Avg score= " + str(average_r2))
#print("Average age diff= " + str(average_age_diff))
print("Complete.")
print("Avg score1= " + str(average_score_1))
print("Avg score2= " + str(average_score_2))
print("Avg score3= " + str(average_score_3))
print("Avg age diff1= " + str(average_age_diff_1))
print("Avg age diff2= " + str(average_age_diff_2))
print("Avg age diff3= " + str(average_age_diff_3))
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

#print("break")
#printWithLines(ages_all)
#print("break")
printWithLines(y_predAll)
#print("break")
#printWithLines(np.abs(ages_all - y_predAll))