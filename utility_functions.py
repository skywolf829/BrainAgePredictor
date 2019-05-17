#Author: Skylar Wurster
#Description: Useful functions called from train.py

import csv
import numpy as np

# Old utility function that would load a CSV and return the
# headers, features, features for group 1, features for group 2, features for group 3, 
# features for all groups, ages, ages for group 1, ages for group 2, ages for group 3, 
# and ages for all groups, in that order, as lists
# Requires that the feature vector starts in column 5 (0 indexed) of the CSV document
# Requires that the group number is in column 1 (0 indexed)
# Requires that the first row is the headers
# Requires that the column 2 is the true age (0 indexed)
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

# Loads a CSV into multiple dictionaries. Returns
# patientToGroup, patientToAge, patientToFeatures, headers, which map patientID
# to group, age, features respectively, and then a simple list of headers.
# Requires that the feature vector starts in column 4 (0 indexed) of the CSV document
# Requires that the patientID is in column 0 (o indexed)
# Requires that the group number is in column 1 (0 indexed)
# Requires that the true age is in column 2 (0 indexed)
# Requires that the first row is the headers
#
# Supports adding multiple CSV file features on to existing dictionaries when multiple_readings=False
# Supports multiple patientID readings when multiple_readings=True
# Does NOT support empty entries in the CSV
# ONLY supports comma delimited CSV
def loadCSVDictionary(filename, patientToGroup = {}, patientToAge = {}, patientToFeatures={}, headers=[], multiple_readings=False):
    currentHeaders = []
    # List of variables that were highly correlated with brain age for selection. Ignored now.
    special = ["superiorfrontal", "GCC", "parsopercularis", "medialorbitofrontal","Thalamus", "superiortemporal","rostralanteriorcingulate","CC", "BCC", "Left-Accumbens-area", "FX", "caudalmiddlefrontal", "insula", "supramarginal", "frontalpole", "rostralmiddlefrontal", "parstriangularis", "bankssts", "CR", "lateralorbitofrontal"]
    # List of variables with high collinearity to remove from selection. Ignored now.
    toIgnore = ["CC", "inferiorparietal", "CR", "IC", "caudalmiddlefrontal", "inferiortemporal", "middletemportal", "superiorfrontal", "superiormarginal", "middletemportal"]
    acceptedCs = []
    # Open the file
    with open(filename) as file:
        csvFile = csv.reader(file)
        r = 0
        # Iterate through each row
        for row in csvFile:
            c = 0
            patientData = []        
            patientID = row[0]
            # iterate through each item in each row
            for item in row:
                # r==0 means first row, meaning headers.
                if r == 0:
                    # Only add the headers for the feature vector
                    if item != None and item != "" and item != " " and c >=4:                        
                        currentHeaders.append(item)
                        headers.append(item)
                        acceptedCs.append(c)
                else:
                    # Make sure it isn't blank data
                    if item != None and item != "" and item != " ":
                        # Add the patient ID to the dictionary if it doesn't exist
                        # with an empty array of features to start
                        if c == 0 and not patientID in patientToFeatures:
                            patientToFeatures[patientID] = []
                        elif c == 1:
                            # Assign the group for the patient
                            patientToGroup[patientID] = int(item)
                        elif c == 2:
                            # Assign the age for the patient
                            patientToAge[patientID] = int(item)
                        elif c >= 4:
                            # Add the feature data to the list of patient data
                            patientData.append(float(item))
                            
                c = c + 1
            # After going through the whole row, add each feature onto the
            # features for the patient
            if len(patientData) > 0 and r != 0:
                if not multiple_readings:
                    patientToFeatures[patientID] = np.ndarray.tolist(np.concatenate((patientToFeatures[patientID], patientData), axis=0))
                else:
                    patientToFeatures[patientID].append(patientData)
            r = r + 1 
    return patientToGroup, patientToAge, patientToFeatures, headers

# Combs through the provided dictionaries and returns the 
# ages and features for the groups requested.
# patientGroups, patientAges, and patientFeatures should be the
# dictionaries returned from loadCSVDictionary
# groups should be a list of groups you'd like returned, such as [0] or [0, 1, 2, 3]
# Returns correctly ordered lists for IDs, groups, ages, and features.
# Supports multiple readings from a single patientID if multiple_readings=True
def getAgesAndFeaturesForGroups(patientGroups, patientAges, patientFeatures, groups, multiple_readings=False):
    patientIDs = []
    patientGroupsList = []
    patientAgesList = []
    patientFeaturesList = []
    patientIDsFull = []
    # Add all the patientIDs to a list that are from the groups requested
    for patient in patientGroups:
        if patientGroups[patient] in groups:
            patientIDs.append(patient)
    # Go through each other dictionary to add the information to the lists returned
    for patientID in patientIDs:        
        if not multiple_readings:
            patientGroupsList.append(patientGroups[patientID])
            patientAgesList.append(patientAges[patientID])
            patientFeaturesList.append(patientFeatures[patientID])
            patientIDsFull.append(patientID)
        else:
            for reading in patientFeatures[patientID]:
                patientGroupsList.append(patientGroups[patientID])
                patientAgesList.append(patientAges[patientID])
                patientFeaturesList.append(reading)
                patientIDsFull.append(patientID)


    return patientIDsFull, patientGroupsList, patientAgesList, patientFeaturesList

# For a 2D array data, returns the columns requested in col. col might
# look like [0], [0, 1, 4, 5], etc.
# Returns the requested columns and does not affect data
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

# Drops cols within dropCols from data and returns a 2D array
# Does not affect data
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

# finds the average difference between arr1 - arr2, element wise
def calcAverageDiff(arr1, arr2):
    diff = 0
    i = 0
    while i < len(arr1):
        diff = diff + arr1[i] - arr2[i]
        i = i + 1
    diff = diff / len(arr1)
    return diff

# finds the absolute average difference abs(arr1 - arr2) element wise
def calcAbsAverageDiff(arr1, arr2):
    diff = 0
    i = 0
    while i < len(arr1):
        diff = diff + abs(arr1[i] - arr2[i])
        i = i + 1
    diff = diff / len(arr1)
    return diff
# prints each item in a 1D array
def printWithLines(arr):
    for item in arr:
        print(item)