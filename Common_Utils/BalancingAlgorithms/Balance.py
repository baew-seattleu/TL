from math import *
from pandas.core.frame import DataFrame
from random import randrange
from scipy import stats
from scipy.stats import gamma
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sn


def calculateRadius(mi, MI_num, ma, MA_num, numAttr, mu=1):
    E = 0
    for i in range(MI_num):
        # Select an instance x in minority class
        x = mi[i]
        x = np.reshape(x, (-1, numAttr))

        # Find distances from minority sample to EVERY majority sample; NOTE: Disregard index output
        dist, _ = findNeighbors(x, ma, MA_num - 1)
        # Calculate sum of distances between minority instance x and ALL majority instances as Ei
        Ei = dist.sum()
        # Accumulate and sum all Ei to get E
        E += Ei

    # Calculate Emean of the sum of distances between all minority samples and majority samples
    Emean = mu * (E / (MI_num * MA_num))
    return Emean


def calculateSampleDensity(value, r, mi, ma):
    mi_nn = NearestNeighbors(radius=r).fit(mi)
    ma_nn = NearestNeighbors(radius=r).fit(ma)
    mi_rng = mi_nn.radius_neighbors(value)
    ma_rng = ma_nn.radius_neighbors(value)
    # Default weight ratio of MA to MI is 8:2
    density = 0.8 * len(np.asarray(ma_rng[1][0])) + 0.2 * len(np.asarray(mi_rng[1][0]))
    return density


def calculateAverageDistance(c1, c1_num, c2, numAttr, k):
    D = 0
    for i in range(c1_num):
        # Select an instance x in minority class
        x = c1[i]
        x = np.reshape(x, (-1, numAttr))

        # Find distances from minority sample to k minority class neighbors; NOTE: Disregard index output
        dist, _ = findNeighbors(x, c2, k)
        Di = dist.sum() / k
        D += Di

    D = D / c1_num
    return D


def BalanceClassesWithGammaDistribution(dataFrame, printDebug=True):
    # Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame["class"] == 1]
    MI = dataFrame[dataFrame["class"] == 0]

    # Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]

    # Calculate how much data to generate
    numToSynthesize = MA_num - MI_num

    # Convert minority class to NumPy array for synthesizing
    mi = MI.to_numpy()
    numAttributes = mi.shape[1]

    # Create NumPy array for synthetic data
    syntheticIndex = 0
    syntheticArray = np.empty((numToSynthesize, numAttributes))

    # Calculate k neighbors to search for
    if (MI_num - 1) < 5:
        k = MI_num - 1
    else:
        k = 5

    if printDebug == True:
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ k = ", k, " ~~~~~~~~~~")

    # default alpha and beta value
    alpha = 0.5
    beta = 0.0125
    maxCD = beta * (alpha - 1)  # coordinate of the max value

    for i in range(numToSynthesize):
        # randomly choosing x in the minority class
        x = random.choice(mi)

        x = np.reshape(x, (-1, numAttributes))

        _, ind = findNeighbors(x, mi, k)

        # Select one knn of sample and record it as y
        y = randrange(1, k + 1)
        x_prime = mi[ind[0, y]]

        # define vector v
        v = x_prime - x

        # generate t using the gamma distribution
        t = stats.gamma.rvs(beta, alpha, random_state=None)
        # new minority point p generated
        p = x + (t - maxCD) * v

        syntheticArray[syntheticIndex] = p
        syntheticIndex += 1

    # convert syntheticArray into DataFrame
    syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)

    # concatenate synthetic dataset with imbalanced dataset
    newDF = pd.concat([dataFrame, syntheticData], ignore_index=True)

    if printDebug:
        print("Number of Majority Instances: ", MA_num)
        print("Number of Minority Instances: ", MI_num)

    print("Here")
    newDF.head(5)
    print("~~~~~~~ Gamma Without BoostCC: ~~~~~~~")
    print("BEFORE BALANCING:\n", newDF["class"].value_counts())

    return newDF


# Find Neighbors
# value - sample to query
# allValues - values to search for neighbors
# k - number of neighbors to search for
def findNeighbors(value, allValues, k):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(allValues)
    # using k+1 since first nearest neighbor is itself
    dist, indices = nn.kneighbors(value, return_distance=True)
    return dist, indices


def calculateControlCoefficient(value, k, mi, ma, Dn, Dp, printDebug=False):
    # Calculate average distance D1 between sample and k minority class neighbors
    dist, _ = findNeighbors(value, mi, k)
    D1 = dist.sum() / k

    # Calculate average distance D2 between sample and k majority class neighbors
    dist, _ = findNeighbors(value, ma, k)
    D2 = dist.sum() / k

    # Calculate the relative distance ratio u according to D1, D2, Dneg, and Dpos

    u = (D1 * Dn) / (D2 * Dp)

    # According to u, calculate control coefficient cc
    if u < 1:
        cc = random.uniform(0, 1)
    elif (u >= 1) & (u <= 2):
        cc = 0.5 + 0.5 * random.uniform(0, 1)
    elif u > 2:
        cc = 0.8 + 0.2 * random.uniform(0, 1)
    if printDebug:
        print("D1 = ", D1)
        print("D2 = ", D2)
        print("u = ", u)
    return cc


def split(x, n):
    arr = []
    if x < n:  # If we cannot split the number into exactly 'N' parts
        print("ERROR")
        return arr
    elif (
        x % n == 0
    ):  # If x % n == 0 then the minimum difference is 0 and all numbers are x/n
        for i in range(n):
            arr.append(x // n)
    else:
        # upto n-(x % n) the values will be x / n
        # values after will be x / n + 1
        zp = n - (x % n)
        pp = x // n
        for i in range(n):
            if i >= zp:
                arr.append(pp + 1)
            else:
                arr.append(pp)
    return arr


def LogisticRegressionCLF(sample_set, org_set):
    from sklearn.model_selection import train_test_split

    x_train = sample_set.values
    y_train = sample_set["class"].values

    x_test = org_set.values
    y_test = org_set["class"].values

    model = LogisticRegression()
    clf = model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    AUC = roc_auc_score(y_test, prediction)

    return AUC


def SMOTEBoostCC(dataFrame, numIterations=5, printDebug=True):
    newDF = dataFrame

    # Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame["class"] == 1]
    MI = dataFrame[dataFrame["class"] == 0]

    # Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]

    # Calculate how much data to generate
    numToSynthesize = MA_num - MI_num

    # Calculate k neighbors to search for
    if (MI_num - 1) < 5:
        k = MI_num - 1
    else:
        k = 5

    # Split numToSynthesize into numIterations for synthesizing
    GENERATE = split(numToSynthesize, numIterations)

    if printDebug:
        print("~~~~~~~~~~~ MA_num = ", MA_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ k = ", k, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ numToSynthesize = ", numToSynthesize, " ~~~~~~~~~~")
        print("GENERATE: ", GENERATE)

    # make copy of MI
    newMI = MI
    newDF = dataFrame

    # Convert majority to NumPy array for synthesizing and density distribution calculations

    ma = MA.to_numpy()

    for synth_num in GENERATE:
        best_recall = 0
        best_set = pd.DataFrame()

        # Convert minority class to NumPy array for synthesizing
        mi = newMI.to_numpy()
        numAttributes = mi.shape[1]

        # Calculate average Euclidean distance Dpos of the total minority class sample set
        Dpos = calculateAverageDistance(mi, MI_num, mi, numAttributes, k)

        # Calculate average Euclidean distance Dneg of the total majority class sample set
        Dneg = calculateAverageDistance(mi, MI_num, ma, numAttributes, k)

        if printDebug:
            print("Dpos = ", Dpos)
            print("Dneg = ", Dneg)

        for __ in range(5):
            # Create NumPy array for synthetic data
            syntheticArray = np.empty((0, numAttributes))

            for i in range(synth_num):
                # Select and instance x in minority class randomly
                x = random.choice(mi)
                # x = mi[np.random.choice(len(mi),replace=False)]
                x = np.reshape(x, (-1, numAttributes))

                controlCoeff = calculateControlCoefficient(x, k, mi, ma, Dneg, Dpos)

                if printDebug:
                    print("cc = ", controlCoeff)

                # Find indices of k nearest neighbors of x
                _, knn = findNeighbors(x, mi, k)

                # Select one knn of sample and record it as y
                y = randrange(1, k + 1)

                # Generate new minority instance w/ equation
                diff = mi[knn[0, y]] - x
                gap = controlCoeff
                xnew = x + gap * diff
                syntheticArray = np.concatenate((syntheticArray, xnew))

            if printDebug:
                print("Iteration Done!")
            # Convert synthetic array into DataFrame
            syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)
            tempDF = pd.concat([newDF, syntheticData], ignore_index=True)

            # Run Logistic Regression on new dataset and original to get recall
            recall = LogisticRegressionCLF(tempDF, dataFrame)
            if recall > best_recall:
                best_recall = recall
                best_set = syntheticData
        if printDebug:
            print("Recall: ", best_recall)
        newMI = pd.concat([newMI, best_set], ignore_index=True)
        newDF = pd.concat([newDF, best_set], ignore_index=True)

    return newDF


def GammaSMOTEBoostCC(dataFrame, numIterations=5, printDebug=False):
    if printDebug:
        print("dataFrame", dataFrame)

    # Calculate number of attributes in dataFrame
    numAttributes = dataFrame.shape[1]

    if printDebug:
        print("numAttributes", numAttributes)

    # Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame["class"] == 1]
    MI = dataFrame[dataFrame["class"] == 0]

    if printDebug:
        print("MA", MA)
        print("MI", MI)

    # Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]

    if printDebug:
        print("MA_num", MA_num)
        print("MI_num", MI_num)

    # Calculate how much data to generate
    numToSynthesize = MA_num - MI_num

    if printDebug:
        print("numToSynthesize", numToSynthesize)

    # Convert minority class to NumPy array for synthesizing
    mi = MI.to_numpy()

    # Apply SMOTEBoostCC to balance the dataset
    balanced_data = SMOTEBoostCC(
        dataFrame, numIterations=numIterations, printDebug=printDebug
    )

    # Convert balanced dataset into NumPy array for synthesizing
    balanced_array = balanced_data.to_numpy()

    # Create NumPy array for synthetic data
    syntheticIndex = 0
    syntheticArray = np.empty((numToSynthesize, numAttributes))

    # Calculate k neighbors to search for
    if (MI_num - 1) < 5:
        k = MI_num - 1
    else:
        k = 5

    # Default alpha and beta value
    alpha = 0.5
    beta = 0.0125
    maxCD = beta * (alpha - 1)  # coordinate of the max value

    generate = numToSynthesize

    while generate > 0:
        # Randomly choosing x in the minority class
        x = random.choice(mi)

        x = np.reshape(x, (-1, numAttributes))
        knn = NearestNeighbors(n_neighbors=k + 1).fit(mi)
        ind = knn.kneighbors(x, return_distance=False)

        # Select one knn of sample and record it as y
        y = randrange(1, k + 1)
        x_prime = mi[ind[0, y]]

        # Define vector v
        v = x_prime - x

        # Generate t using the gamma distribution
        t = stats.gamma.rvs(beta, alpha, random_state=None)

        p = x + (t - maxCD) * v

        syntheticArray[syntheticIndex] = p
        syntheticIndex += 1
        generate -= 1

    # Convert syntheticArray into DataFrame
    syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)

    # Concatenate synthetic dataset with imbalanced dataset
    newDF = pd.concat([balanced_data, syntheticData], ignore_index=True)

    return newDF


def generate_autoencoder(original_df, class_var, minority_var, printDebug=False):
    # Divide data set into Majority (MA) and Minority (MI) Classes

    print("Before balancing count\n", original_df[class_var].value_counts())
    if not minority_var:
        minority_var = 0

    majority_df = original_df[original_df["class"] != minority_var]
    minority_df = original_df[original_df["class"] == minority_var]

    # Save the Minority Class for later
    MI_saved = minority_df.copy()
    # minority_df = minority_df.drop(columns=['class'])
    input_shape = MI_saved.shape[1]

    default_class = MI_saved["class"].iloc[0]
    MI_saved.drop(columns=["class"], inplace=True)

    MI_saved = MI_saved.astype(np.float32)
    MI_saved = MI_saved.sample(frac=1, random_state=42).reset_index(drop=True)

    MI_num = MI_saved.shape[0]
    input_shape = MI_saved.shape[1]
    numToSynthesize = MI_num * 2

    if numToSynthesize < 0:
        numToSynthesize = 50

    input_shape = int(input_shape)
    if input_shape < 0:
        raise ValueError("Input shape must be greater than 0.")

    autoencoder = tensorflow.keras.models.load_model(
        "balanced_autoencoder.h5", compile=False
    )

    class_count_diff = majority_df.shape[0] - minority_df.shape[0]

    generated_data = pd.DataFrame()

    # balance the data
    while generated_data.shape[0] < class_count_diff:
        generated_samples = autoencoder.predict(
            MI_saved, verbose=0
        )  # Generate synthetic samples
        generated_data = pd.concat(
            [generated_data, pd.DataFrame(generated_samples, columns=MI_saved.columns)],
            ignore_index=True,
        )

    generated_data = generated_data[:class_count_diff]

    df_generated = pd.DataFrame(generated_data, columns=MI_saved.columns)
    df_generated["class"] = default_class
    newDF = pd.concat([df_generated, minority_df, majority_df], ignore_index=True)
    shuffled_df = newDF.sample(frac=1, random_state=42)

    return shuffled_df


def SDDSMOTEBoostCC(dataFrame, printDebug=True):
    # Calculate number of attributes in dataFrame
    numAttributes = dataFrame.shape[1]

    # Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame["class"] == 1]
    MI = dataFrame[dataFrame["class"] == 0]

    # Record number of instances for data set and each class
    DF_num = dataFrame.shape[0]
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]

    # Calculate number of data samples to generate
    numToSynthesize = MA_num - MI_num

    # Calculate k-nearest neighbors to search for
    if (MI_num - 1) < 5:
        k = MI_num - 1
    else:
        k = 5

    if printDebug:
        print("~~~~~~~~~~~ DF_num = ", DF_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ MA_num = ", MA_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ k = ", k, " ~~~~~~~~~~")

    # Convert dataframe to NumPy array for synthesizing and density distribution calculations
    df = dataFrame.to_numpy()
    mi = MI.to_numpy()
    ma = MA.to_numpy()

    # Create NumPy array for synthetic data
    syntheticIndex = 0
    syntheticArray = np.empty((numToSynthesize, numAttributes))
    # ************************************************************************************
    # Calculate radius to delimit circular area for density calculations
    radius = calculateRadius(mi, MI_num, ma, MA_num, numAttributes, mu=2)

    # Create Density List of all minority samples
    densityList = []
    for i in range(MI_num):
        sample = mi[i]
        sample = np.reshape(sample, (-1, numAttributes))

        # Calculate density D of each minority class sample
        D = calculateSampleDensity(sample, radius, mi, ma)

        densityList.append((i, D))
    densityList.sort(key=lambda x: x[1], reverse=True)

    # Calculate average Euclidean distance Dpos of the total minority class sample set
    Dpos = calculateAverageDistance(mi, MI_num, mi, numAttributes, k)

    # Calculate average Euclidean distance Dneg of the total majority class sample set
    Dneg = calculateAverageDistance(mi, MI_num, ma, numAttributes, k)

    if printDebug:
        print("Radius = ", radius)
        print("densityList = ", densityList)
        print("Dpos = ", Dpos)
        print("Dneg = ", Dneg)

    densityIndex = 0
    for i in range(numToSynthesize):
        if densityIndex >= len(densityList):
            densityIndex = 0

        # Based chose sample based on density from densityList
        instance = densityList[densityIndex]
        index = instance[0]
        x = mi[index]
        x = np.reshape(x, (-1, numAttributes))

        # Calculate Control Coefficient
        cc = calculateControlCoefficient(x, k, mi, ma, Dneg, Dpos)

        if printDebug:
            print("cc = ", cc)

        # Find indices of k nearest neighbors of x; NOTE: Disregard distance output
        _, knn = findNeighbors(x, mi, k)

        # Select one knn of sample and record it as y
        y = randrange(1, k + 1)

        # Generate new minority instance w/ equation
        diff = mi[knn[0, y]] - x
        syntheticArray[syntheticIndex] = x + cc * diff
        syntheticIndex += 1
        densityIndex += 1

    # Convert synthetic array into DataFrame
    syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)

    # Join imbalance data with synthetic data set
    newDF = pd.concat([dataFrame, syntheticData], ignore_index=True)

    return newDF
