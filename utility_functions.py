import csv
import numpy as np

def loadCSV(filename):
    grouping = 10
    features = []
    features_1 = []
    features_2 = []
    features_3 = []
    features_all = []
    age_groups = []
    age_groups_1 = []
    age_groups_2 = []
    age_groups_3 = []
    ages = []
    ages_1 = []
    ages_2 = []
    ages_3 = []
    ages_all = []
    headers = []
    with open(filename) as file:
        csvFile = csv.reader(file)
        r = 0
        for row in csvFile:
            control = 0
            c = 0
            patientData = []        
            for item in row:
                if r == 0:
                    if item != None and item != "" and item != " " and c >= 5:
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
                        if c >= 5:
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
    
    features_all = np.concatenate((features, features_1, features_2, features_3))
    ages_all = np.concatenate((ages, ages_1, ages_2, ages_3))

    return headers, features, features_1, features_2, features_3, features_all, ages, ages_1, ages_2, ages_3, ages_all

def loadCSVDictionary(filename, patientToGroup = {}, patientToAge = {}, patientToFeatures={}, headers=[]):
    currentHeaders = []
    special = ["superiorfrontal", "GCC", "parsopercularis", "medialorbitofrontal","Thalamus", "superiortemporal","rostralanteriorcingulate","CC", "BCC", "Left-Accumbens-area", "FX", "caudalmiddlefrontal", "insula", "supramarginal", "frontalpole", "rostralmiddlefrontal", "parstriangularis", "bankssts", "CR", "lateralorbitofrontal"]
    
    with open(filename) as file:
        csvFile = csv.reader(file)
        r = 0
        for row in csvFile:
            c = 0
            patientData = []        
            for item in row:
                if r == 0:
                    if item != None and item != "" and item != " ":
                        currentHeaders.append(item)
                        if c >= 4:
                            headers.append(item)
                else:
                    if item != None and item != "" and item != " ":
                        if c == 0 and not row[0] in patientToFeatures:
                            patientToFeatures[row[0]] = []
                        elif c == 1:
                            patientToGroup[row[0]] = int(item)
                        elif c == 2:
                            patientToAge[row[0]] = int(item)
                        elif c >= 4:
                            patientData.append(float(item))
                            
                c = c + 1
            if len(patientData) > 0 and r != 0:
                patientToFeatures[row[0]] = np.ndarray.tolist(np.concatenate((patientToFeatures[row[0]], patientData), axis=0))
            r = r + 1 
    return patientToGroup, patientToAge, patientToFeatures, headers

def getAgesAndFeaturesForGroups(patientGroups, patientAges, patientFeatures, groups):
    patientIDs = []
    patientGroupsList = []
    patientAgesList = []
    patientFeaturesList = []

    for patient in patientGroups:
        if patientGroups[patient] in groups:
            patientIDs.append(patient)

    for patientID in patientIDs:
        patientGroupsList.append(patientGroups[patientID])
        patientAgesList.append(patientAges[patientID])
        patientFeaturesList.append(patientFeatures[patientID])

    return patientIDs, patientGroupsList, patientAgesList, patientFeaturesList

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
        diff = diff + arr1[i] - arr2[i]
        i = i + 1
    diff = diff / len(arr1)
    return diff

def calcAbsAverageDiff(arr1, arr2):
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