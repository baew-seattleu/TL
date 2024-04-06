# Import Necessary Libraries
import random
import pandas as pd
import numpy as np
from random import randrange

# For data analysis
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import stats
from pandas.core.frame import DataFrame
from scipy.stats import gamma
from sklearn.neighbors import NearestNeighbors
from math import *
import math


# Find Neighbors
# value - sample to query
# allValues - values to search for neighbors
# k - number of neighbors to search for
def findNeighbors(value, allValues, k):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(allValues)
    # using k+1 since first nearest neighbor is itself
    dist, indices = nn.kneighbors(value, return_distance=True)
    return dist, indices


# AD-KNN Algorithm
# Balances an imbalanced binary classification dataset using Average Difference KNN.
#  dataFrame - the binary classification dataset to balance (dataFrame)
#  classIndex - the name of the column containing class labels
#  minorityLabel - the value in the classIndex column that indicates the instance is a minority, such as 0
#  kMin - minimum k-value to use when generating a random k-value for the KNN search
#  kMax - maximum k-value to use when generating a random k-value for the KNN search
#  epsilon - when the average difference is 0 (i.e., the generated data point is a duplicate of the existing
#                  data point), epsilon is how far away from the existing point to place the new point


def ADKNN(
    dataFrame,
    classIndex,
    minorityLabel,
    kMin=2,
    kMax=6,
    printDebug=False,
    epsilon=0.0001,
):
    # Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame[classIndex] != minorityLabel]
    MI = dataFrame[dataFrame[classIndex] == minorityLabel]

    # Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]

    # Calculate how much data to generate
    numToSynthesize = MA_num - MI_num

    # Make sure kMax has a valid value (does not exceed the number of minority instances)
    if (MI_num - 1) < kMax:
        kMax = MI_num - 1

    # Make sure kMin and kMax are not the same in order not to generate duplicates.
    # If so, set kMin to a default value (kMax / 2).
    if kMin >= kMax:
        kMin = int(np.ceil(kMax / 2))

    if printDebug:
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ MA_num = ", MA_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ kMin = ", kMin, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ kMax = ", kMax, " ~~~~~~~~~~")

    # Convert minority class to NumPy array for synthesizing
    mi = MI.to_numpy()

    # Number of attributes for each data instance
    numAttributes = mi.shape[1]

    # Create NumPy array that holds the synthetic data
    syntheticIndex = 0
    syntheticArray = np.empty((numToSynthesize, numAttributes))

    # Stores the average of the differences between a query point and its nearest neighbors
    averageDifference = np.zeros(numAttributes)

    # The position/index number of the class column label
    classColNum = dataFrame.columns.get_loc(key=classIndex)

    # Dictionary that stores the list of k-values that have already been used by each minority
    # point to generate data, in order to avoid generating duplicates
    pointDict = {}

    # Initialize each point to have the value of an empty list. k-values will be added to the list
    # Convert each point (a NumPy array) to a tuple to avoid an error, since a NumPy array is unhashable.
    for point in mi:
        pointDict[tuple(point)] = []

    for i in range(numToSynthesize):
        # Flag that stops the loop if all possible data points have been generated
        done = True

        # Check if all possible data points have been generated (if all the minority points have used all the k-values)
        for val in pointDict.values():
            if len(val) < kMax - kMin + 1:
                done = False

        # If there are still data points left to generate, generate one
        if not done:
            # Select an instance in the minority class randomly
            point = random.choice(mi)

            # Convert point to a tuple - this is the dictionary key corresponding to point
            t = tuple(point)

            # Make sure this minority point has not used all the possible k-values.
            # If so, pick a new one.
            while len(pointDict[t]) == kMax - kMin + 1:
                point = random.choice(mi)
                t = tuple(point)

            point = np.reshape(point, (-1, numAttributes))

            # Generate the random k-value used for the KNN search in this iteration.
            k = random.randint(kMin, kMax)

            # If this k-value has already been used for this point, then generate a
            # new k-value that hasn't been used
            while k in pointDict[t]:
                k = random.randint(kMin, kMax)

            # Append the newly generated k-value to the list of used k-values for this point
            pointDict[t].append(k)

            # Find indices of k nearest neighbors of point. "_" is used to discard the distances to the neighbors.
            _, neighborIndices = findNeighbors(point, mi, k)

            point = point.flatten()

            # Discard the first neighbor returned, because the first neighbor is always
            # the query point itself, and flatten neighborIndices into a 1D array
            neighborIndices = neighborIndices[0][1::]

            # Sum all the differences between the query point and its neighbors, to find their average
            for j in range(len(neighborIndices)):
                # The data point that is the jth neighbor of point
                neighbor = mi[neighborIndices[j]]

                # Difference/displacement between the query point and neighbor
                diff = neighbor - point

                # Add diff to the sum of all the differences
                averageDifference = np.add(averageDifference, diff)

            # Find the average by dividing the sum of the differences by k
            averageDifference = np.divide(averageDifference, float(k))

            # New point is the query point plus the average difference between the query point and its neighbors
            newPoint = np.add(point, averageDifference)

            # If averageDifference is 0, a duplicate will be generated, so add epsilon to
            # all attributes of averageDifference
            if np.array_equal(averageDifference, np.zeros(numAttributes)):
                newPoint = np.add(newPoint, np.full(numAttributes, epsilon))

            syntheticArray[syntheticIndex] = newPoint

            # Make sure the class label of the new data point is the minority class label
            syntheticArray[syntheticIndex][classColNum] = minorityLabel

            averageDifference = np.zeros(
                numAttributes
            )  # reset all elements to 0 for next iteration
            syntheticIndex += 1

    # If the number of points generated is less than numToSynthesize, resize syntheticArray
    # to that number of points, to avoid garbage values.
    if syntheticIndex < numToSynthesize:
        syntheticArray = syntheticArray[0:syntheticIndex]

    # Convert synthetic array into DataFrame
    syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)

    # Join imbalanced data with synthetic data set
    newDF = pd.concat([dataFrame, syntheticData], ignore_index=True)

    if printDebug:
        print(
            "~~~~~~~ Number of points generated:", syntheticArray.shape[0], " ~~~~~~~"
        )
        print("~~~~~~~ Length of balanced dataset:", newDF.shape[0], " ~~~~~~~")

    return newDF
