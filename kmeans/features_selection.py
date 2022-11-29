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
        if(i > 50):
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


# def nnAlg(trainX, trainY, featuresIdxs):
#     predY = []
#     trainXTemp = trainX[:, featuresIdxs]
#     for k in range(0, len(trainX)):
#         sampleX = np.array([trainX[k]])
#         sampleXTemp = sampleX[:, featuresIdxs]
#         eucDises = np.sum(np.abs(trainXTemp - sampleXTemp), axis=1)
#         # eucDises = np.sqrt(np.sum(np.square(trainXTemp - sampleXTemp), axis=1))
#         sortedEucDises = np.argsort(eucDises)
#         predY.append(trainY[sortedEucDises[1]])
#     return predY

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

def selection(trainX, trainY):
    bestLvlFeatureList = []
    bestUpperLvlFeatureList = []
    bestLowerLvlFeatureList = []
    bestFeatureList = []
    (m, n) = trainX.shape
    for j in range(0, n):
        # if (j > 20):
        #     break
        lvlErrRate = 1
        for i in range(0, n):
            if (featureInList(i, bestUpperLvlFeatureList)):
                continue
            tempFeatureList = bestUpperLvlFeatureList[:]
            tempFeatureList.append(i)
            errRate = errorRate(trainY, nnAlg(trainX, trainY, tempFeatureList))
            # print("    Using feature(s) " + str(
            #     np.sort(tempFeatureList, kind='quicksort', order=None)) + " accuracy is " + str(
            #     (1 - errRate) * 100.0) + "%")
            if (errRate < lvlErrRate):
                lvlErrRate = errRate
                bestLvlFeatureList = tempFeatureList
        print("Feature set #"+ str(j) + ":" + str(
            np.sort(bestLvlFeatureList, kind='quicksort', order=None)) + " Accuracy: " + str(
            (1 - lvlErrRate) * 100) + "%")
        bestUpperLvlFeatureList = bestLvlFeatureList
        bestFeatureList.insert(0, [lvlErrRate, np.sort(bestLvlFeatureList, kind='quicksort', order=None)])
    errRate = 1
    idx = -1
    for i in range(0, len(bestFeatureList)):
        tempFeatureList = bestFeatureList[i]
        if (errRate > tempFeatureList[0]):
            errRate = tempFeatureList[0]
            idx = i
    print("The best featrue set is " + str(bestFeatureList[idx][1]) + " Accuracy: " + str(
        (1 - bestFeatureList[idx][0]) * 100) + "%")
    return np.array(bestFeatureList, dtype=object)

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
    imgNames, imgPreds, imgValues, imgFeatures = loadData("classifier_feature_record.txt")
    start = time.time()
    bestFeatureList = selection(imgFeatures, imgPreds)
    end = time.time()
    print("time:" + str(end - start))
    plotChart(bestFeatureList)
    print("done\n")