import pandas as pd

# Balances the classes in the dataframe at the desired index by duplicating entries for that index at random
# @param dataFrame - The dataFrame to add duplicates to
# @param index - The index that contains the class


def BalanceClassesWithDuplicates(dataFrame, index, printDebug=True):
    newDF = dataFrame
    classes = dataFrame[index].unique()
    numberOfClasses = len(dataFrame[index].unique())
    if printDebug:
        print("classes: [" + "".join(f"{str(e)}," for e in classes) + "]")
        print(f"Number of Classes:{numberOfClasses}")

    sizeOfClasses = [
        len(dataFrame.values[dataFrame[index].values == aClass]) for aClass in classes
    ]
    sizeOfLargestClass = max(sizeOfClasses)

    if printDebug:
        print(
            "Size of Classes: [" + "".join(f"{str(e)}, " for e in sizeOfClasses) + "]"
        )
        print(f"Size of Largest Class: {str(sizeOfLargestClass)}")

    amountToAddToEachClass = [
        sizeOfLargestClass - classSize for classSize in sizeOfClasses
    ]
    if printDebug:
        print(
            "Amount to Add to Each Class: ["
            + "".join(f"{str(e)}, " for e in amountToAddToEachClass)
            + "]"
        )

    for k, amount in enumerate(amountToAddToEachClass):
        for _ in range(amount):
            # newDF = newDF.append(dataFrame[dataFrame[index].values == classes[k]].sample(n=1, replace=True))
            newDF = pd.concat(
                [
                    newDF,
                    dataFrame[dataFrame[index].values == classes[k]].sample(
                        n=1, replace=True
                    ),
                ]
            )
    return newDF
