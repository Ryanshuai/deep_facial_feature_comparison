import numpy as np
import matplotlib.pyplot as plt
import time

def loadData(fileName):
    imgNames = []
    imgPreds = []
    imgValues = []
    imgFeatures = []
    file = open(fileName, "r")
    i = 0
    while True:
        i+=1
        if(i > 5000):
            break
        line = file.readline()
        if not line:
            break
        dataLine = []
        dataStr = line.split("  ")
        imgNames.append(dataStr[0])
        imgPreds.append(int(dataStr[1]))
        imgValues.append(float(dataStr[2]))
        featruesStr = dataStr[3].strip().strip('[]')
        # featruesStr.replace(' ', '')
        strFeatures = featruesStr.split(', ')
        features = []
        for feature in strFeatures:
            features.append(float(feature))
        imgFeatures.append(features)

    return imgNames, np.array(imgPreds, dtype=np.int8), np.array(imgValues, dtype=np.float64), np.array(imgFeatures, dtype=np.float64)

def varianceSelection(trainX, trainY):
    selectedFeatures = []
    (m, n) = trainX.shape
    variances = []
    for i in range(0, n):
        sampleX = trainX[:, i]
        mean = sampleX.mean()
        # variance = np.sqrt(np.sum(np.square(sampleX - mean), axis=0)/(m - 1))
        # v1 = sampleX - mean
        # v2 = np.square(sampleX - mean)
        # v3 = np.sum(np.square(sampleX - mean), axis=0)
        variance = np.sum(np.square(sampleX - mean), axis=0)/(m - 1)
        variances.append(variance)
    sortedVariances = np.argsort(np.array(variances))
    threshhold = 50.0
    for i in range(0, n):
        if(variances[sortedVariances[i]] >= threshhold):
            selectedFeatures.append(sortedVariances[i])
            print("variance " + str(sortedVariances[i]) + ": " + str(variances[sortedVariances[i]]) + "\n")
    print("varianceSelection selectedFeatures are " + str(selectedFeatures) + "\n")
    return selectedFeatures

def pearsonCorrelationSelection(trainX, trainY):
    selectedFeatures = []
    (m, n) = trainX.shape
    meanY = trainY.mean()
    rs = []
    for i in range(0, n):
        sampleX = trainX[:, i]
        meanX = sampleX.mean()
        numerator = np.dot((sampleX - meanX), (trainY - meanY))
        denominator = np.sum(np.square(sampleX - meanX), axis=0)*np.sum(np.square(trainY - meanY), axis=0)
        r = numerator/denominator
        rs.append(r)
    sortedRs = np.argsort(np.array(rs))
    k = 10
    for i in range(0, k):
        print("pearsonr " + str(sortedRs[n - i - 1]) + ": " + str(rs[sortedRs[n - i - 1]]) + "\n")
        selectedFeatures.append(sortedRs[n - i - 1])
    return selectedFeatures

def fisherScoreSelection(trainX, trainY):
    selectedFeatures = []
    (m, n) = trainX.shape
    trainX0 = []
    trainX1 = []
    for j in range(0, m):
        if(trainY[j] == 0):
            trainX0.append(trainX[j])
        else:
            trainX1.append(trainX[j])
    trainX0 = np.array(trainX0)
    trainX1 = np.array(trainX1)
    scores = []
    for i in range(0, n):
        sampleX = trainX[:, i]
        sampleX0 = trainX0[:, i]
        sampleX1 = trainX1[:, i]
        mean = sampleX.mean()
        mean0 = sampleX0.mean()
        mean1 = sampleX1.mean()
        m0 = sampleX0.size
        m1 = sampleX1.size
        # v0 = sampleX0 - mean0
        v0 = np.sum(np.square(sampleX0 - mean0), axis=0)/(m0 - 1)
        v1 = np.sum(np.square(sampleX1 - mean1), axis=0)/(m1 - 1)
        numerator = m0*np.square(mean0 - mean) + m1*np.square(mean1 - mean)
        denominator = m0*np.square(v0) + m1*np.square(v1)
        score = numerator / denominator
        scores.append(score)
    sortedScores = np.argsort(np.array(scores))
    score = 1.0
    for i in range(0, n):
        if(scores[sortedScores[n - i - 1]] >= 1.0):
            print("fisherScoreSelection " + str(sortedScores[n - i - 1]) + ": " + str(scores[sortedScores[n - i - 1]]) + "\n")
            selectedFeatures.append(sortedScores[n - i - 1])

    return selectedFeatures

def forwardSelection(trainX, trainY):
    return selection(trainX, trainY, 1)

def backwardSelection(trainX, trainY):
    return selection(trainX, trainY, 2)

def nnAlg(trainX, trainY, featuresIdxs):
    predY = []
    trainXTemp = trainX[:, featuresIdxs]
    for k in range(0, len(trainX)):
        sampleX = np.array([trainX[k]])
        sampleXTemp = sampleX[:, featuresIdxs]
        eucDises = np.sum(np.abs(trainXTemp - sampleXTemp), axis=1)
        # eucDises = np.sqrt(np.sum(np.square(trainXTemp - sampleXTemp), axis=1))
        sortedEucDises = np.argsort(eucDises)
        vote1 = 0
        vote2 = 0
        #k nn using k=5
        for i in range(1, 6):
            if(trainY[sortedEucDises[i]] == 0):
                vote1 += 1
            else:
                vote2 += 1
        if(vote1 > vote2):
            predY.append(0)
        else:
            predY.append(1)
        # predY.append(trainY[sortedEucDises[1]])
    return predY

def errorRate(Y, predy):
    return (predy != Y).mean()

def featureInList(feature, featureList):
    for i in range(0, len(featureList)):
        if (feature == featureList[i]):
            return True
    return False

def selection(trainX, trainY, dirction):
    bestLvlFeatureList = []
    bestUpperLvlFeatureList = []
    bestLowerLvlFeatureList = []
    bestFeatureList = []
    (m, n) = trainX.shape
    if (dirction == 1):
        for j in range(0, n):
            lvlErrRate = 1
            for i in range(0, n):
                if (featureInList(i, bestUpperLvlFeatureList)):
                    continue
                tempFeatureList = bestUpperLvlFeatureList[:]
                tempFeatureList.append(i)
                errRate = errorRate(trainY, nnAlg(trainX, trainY, tempFeatureList))
                print("    Using feature(s) " + str(
                    np.sort(tempFeatureList, kind='quicksort', order=None)) + " accuracy is " + str(
                    (1 - errRate) * 100) + "%")
                if (errRate < lvlErrRate):
                    lvlErrRate = errRate
                    bestLvlFeatureList = tempFeatureList
            print("Best forward feature set is " + str(
                np.sort(bestLvlFeatureList, kind='quicksort', order=None)) + " accuracy is " + str((1 - lvlErrRate) * 100) + "%")
            bestUpperLvlFeatureList = bestLvlFeatureList
            bestFeatureList.insert(0, [lvlErrRate, np.sort(bestLvlFeatureList, kind='quicksort', order=None)])
    else:
        for i in range(0, n):
            bestLowerLvlFeatureList.append(i)
        lvlErrRate = errorRate(trainY, nnAlg(trainX, trainY,
                                             bestLowerLvlFeatureList))  # Calculate the error rate with all features
        bestFeatureList.append([lvlErrRate, bestLowerLvlFeatureList])
        for j in range(0, n - 1):
            lvlErrRate = 1
            for i in range(0, (n - j)):
                tempFeatureList = bestLowerLvlFeatureList[:]
                del tempFeatureList[i]
                errRate = errorRate(trainY, nnAlg(trainX, trainY, tempFeatureList))
                print("    Using feature(s) " + str(tempFeatureList) + " accuracy is " + str((1 - errRate) * 100) + "%")
                if (errRate < lvlErrRate):
                    lvlErrRate = errRate
                    bestLvlFeatureList = tempFeatureList
            print("Best backward feature set is " + str(bestLvlFeatureList) + " accuracy is " + str(
                (1 - lvlErrRate) * 100) + "%")
            bestLowerLvlFeatureList = bestLvlFeatureList
            bestFeatureList.insert(0, [lvlErrRate, bestLvlFeatureList])
    errRate = 1
    idx = -1
    for i in range(0, len(bestFeatureList)):
        tempFeatureList = bestFeatureList[i]
        if (errRate > tempFeatureList[0]):
            errRate = tempFeatureList[0]
            idx = i
    print("Finish searchinng. The best featrue set is " + str(bestFeatureList[idx][1]) + " accuracy is " + str(
        (1 - bestFeatureList[idx][0]) * 100) + "%")
    return np.array(bestFeatureList, dtype=object)

# def selection(trainX, trainY, direction):
#     bestLvlFeatureList = []
#     bestUpperLvlFeatureList = []
#     bestLowerLvlFeatureList = []
#     bestFeatureList = []
#     (m, n) = trainX.shape
#     for j in range(0, n):
#         # if (j > 20):
#         #     break
#         lvlErrRate = 1
#         for i in range(0, n):
#             if (featureInList(i, bestUpperLvlFeatureList)):
#                 continue
#             tempFeatureList = bestUpperLvlFeatureList[:]
#             tempFeatureList.append(i)
#             errRate = errorRate(trainY, nnAlg(trainX, trainY, tempFeatureList))
#             # print("    Using feature(s) " + str(
#             #     np.sort(tempFeatureList, kind='quicksort', order=None)) + " accuracy is " + str(
#             #     (1 - errRate) * 100.0) + "%")
#             if (errRate < lvlErrRate):
#                 lvlErrRate = errRate
#                 bestLvlFeatureList = tempFeatureList
#         print("Feature set #"+ str(j) + ":" + str(
#             np.sort(bestLvlFeatureList, kind='quicksort', order=None)) + " Accuracy: " + str(
#             (1 - lvlErrRate) * 100) + "%")
#         bestUpperLvlFeatureList = bestLvlFeatureList
#         bestFeatureList.insert(0, [lvlErrRate, np.sort(bestLvlFeatureList, kind='quicksort', order=None)])
#     errRate = 1
#     idx = -1
#     for i in range(0, len(bestFeatureList)):
#         tempFeatureList = bestFeatureList[i]
#         if (errRate > tempFeatureList[0]):
#             errRate = tempFeatureList[0]
#             idx = i
#     print("The best featrue set is " + str(bestFeatureList[idx][1]) + " Accuracy: " + str(
#         (1 - bestFeatureList[idx][0]) * 100) + "%")
#     return np.array(bestFeatureList, dtype=object)

def plotChart(bestFeatureList):
    x_data = []
    y_data = []
    title = ""

    title += "Forward Selection"

    for i in range(0, len(bestFeatureList)):
        x_data.append(i)
        y_data.append(100 * (1 - bestFeatureList[i][0]))

    fs = 5
    plt.figure(figsize=(15, 10))
    bars = plt.barh(x_data, y_data, fc='y')

    # for i in range(0, len(bars)):
    #     bar = bars[i]
    #     height = bar.get_height()
    #     text = str('%2.2f' % (y_data[i])) + '%'
    #     plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, text, fontsize=fs)
        # fl = []
        # for j in range(0, len(bestFeatureList[i][1])):
        #     fl.append(bestFeatureList[i][1][j])
        # plt.text(1, bar.get_y() + bar.get_height() / 2, "Set #" + str(i), fontsize=fs)
    plt.xticks([])
    # plt.yticks([])
    plt.title(title, fontsize=20)
    plt.xlabel("Accuracy", fontsize=20)
    plt.ylabel("Feature Set #", fontsize=20)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    imgNames, imgPreds, imgValues, imgFeatures = loadData("/Users/cxliu/Documents/Code/CS235/deep_facial_feature_comparison/feature_selection/classifier_feature_record.txt")
    start = time.time()
    # varianceSelection(imgFeatures, imgPreds)
    # pearsonCorrelationSelection(imgFeatures, imgPreds)
    # fisherScoreSelection(imgFeatures, imgPreds)
    # forwardSelection(imgFeatures, imgPreds)
    backwardSelection(imgFeatures, imgPreds)
    end = time.time()
    print("time:" + str(end - start))
    # plotChart(bestFeatureList)
    print("done\n")