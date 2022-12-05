import numpy as np
import random 
import matplotlib.pyplot as plt 
import time
import sys
sys.path.append('../features_selection')
import features_selection
from features_selection import varianceSelection
from features_selection import pearsonCorrelationSelection
from features_selection import fisherScoreSelection
from features_selection import forwardSelection
from features_selection import backwardSelection
from features_selection import errorRate

def loadData(fileName):
    imgNames = []
    imgPreds = []
    imgValues = []
    imgFeatures = []
    file = open(fileName, "r")
    i = 0
    while True:
        i+=1
        # if(i > 100):
        #     break
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

def manhattanDistance(data, center):
    # dises = np.linalg.norm(data - center, ord=1)
    dises = np.sum(np.abs(data - center), axis=1)
    return dises

def manhattanP2PDistance(data, center):
    dis = np.sum(np.abs(data - center))
    return dis

def euclideanDistance(data, center):
    dises = np.sqrt(np.sum(np.square(data - center), axis=1))
    return dises

def euclideanP2PDistance(data, center):
    dis = np.sqrt(np.sum(np.square(data - center)))
    return dis

def chebyshevDistance(data, center):
    v1 = data - center
    v2 = np.abs(v1)
    dises = np.max(np.abs((data - center)), axis=1)
    # dises = np.abs((data - center)).max()
    return dises

def chebyshevP2PDistance(data, center):
    dises = np.abs(data - center).max()
    return dises

def chebyshevDistance2(data, center):
    temp = data - center
    dises = np.linalg.norm(data - center, ord=np.inf, axis=1)
    # dises2 = np.linalg.norm(data[0] - center, ord=np.inf)
    return dises

def chebyshevP2PDistance2(data, center):
    temp = data - center
    dises = np.linalg.norm(data - center, ord=np.inf)
    return dises

def cosineDistance(data, center):
    # v1 = np.dot(data,center)
    # v11 = np.linalg.norm(data, axis=1)
    # v2 = np.dot(data[0],center)
    # v22 = np.linalg.norm(data[0])
    dises = np.dot(data,center)/(np.linalg.norm(data, axis=1)*(np.linalg.norm(center)))
    # dises2 = np.dot(data[0],center)/(np.linalg.norm(data[0])*(np.linalg.norm(center)))
    return dises

def cosineP2PDistance(data, center):
    dises = np.dot(data,center)/(np.linalg.norm(data)*(np.linalg.norm(center)))
    return dises

# def distance2(point1, point2):
#     return np.sqrt(np.sum(np.square(point1 - point2)))

def k_means(data, trainY, k, disFunc, max_iter=300):
    cs = []
    n = data.shape[0]
    loop = True
    i = 0
    while(loop):
        i += 1
        cs = []
        for e in enumerate(random.sample(range(n), k)):
            if(e[0] == 0):
                print("0 i = " + str(i) + " e[1]: " + str(e[1]) + " trainY[e[1]]: " + str(trainY[e[1]]))
                if(trainY[e[1]] != 0):
                    break
            if(e[0] == 1):
                print("0 i = " + str(i) + " e[1]: " + str(e[1]) + " trainY[e[1]] :" + str(trainY[e[1]]))
                if(trainY[e[1]] != 1):
                    break
                loop = False
            cs.append(data[e[1]])

    for i in range(max_iter):
        print("k_means disFunc:" + str(disFunc) + " iteration {}".format(i+1))
        preds = []
        clusters = {}
        for j in range(k):
            clusters[j] = []

        eucDisesList = []
        for j in range(k):
            # eucDises = distance(data, data[0])
            if(disFunc == 0):
                eucDises = manhattanDistance(data, cs[j])
            elif(disFunc == 1):
                eucDises = euclideanDistance(data, cs[j])
            elif(disFunc == 2):
                eucDises = chebyshevDistance(data, cs[j])
            elif(disFunc == 3):
                eucDises = chebyshevDistance2(data, cs[j])
            elif(disFunc == 4):
                eucDises = cosineDistance(data, cs[j])
            eucDisesList.append(eucDises)

        for j in range(n):
            minDis = eucDisesList[0][j]
            clusterIdx = 0
            m = 1
            while m < k:
                dis = eucDisesList[m][j]
                if(disFunc == 4):
                    if(dis > minDis):
                        minDis = dis
                        clusterIdx = m
                else:
                    if(dis < minDis):
                        minDis = dis
                        clusterIdx = m
                m += 1
            preds.append(clusterIdx)
            clusters[clusterIdx].append(data[j])
            
        precs = cs.copy()

        for c in clusters.keys():
            cs[c] = np.mean(clusters[c], axis=0)
  
        conv = True
        for i in range(k):
            # dis2 = manhattanP2PDistance(precs[i], cs[i])
            if(disFunc == 0):
                eucDises = manhattanP2PDistance(precs[i], cs[i])
            elif(disFunc == 1):
                eucDises = euclideanP2PDistance(precs[i], cs[i])
            elif(disFunc == 2):
                eucDises = chebyshevP2PDistance(precs[i], cs[i])
            elif(disFunc == 3):
                eucDises = chebyshevP2PDistance2(precs[i], cs[i])
            elif(disFunc == 4):
                eucDises = cosineP2PDistance(precs[i], cs[i])
            print("Center " +str(i) + " dis2:" + str(eucDises))
            if(disFunc == 4):
                if eucDises < 1 - 1e-10:
                    conv = False
                    break
            else:
                if eucDises > 1e-10:
                    conv = False
                    break
        if conv == True:  
            break
    err = errorRate(trainY, preds)
    print("k_means disFunc:" + str(disFunc) + " errorRate:" + str(err))
    return cs, clusters, preds

if __name__ == '__main__':
    # imgNames, imgPreds, imgValues, imgFeatures = loadData("classifier_feature_record.txt")
    imgNames, imgPreds, imgValues, imgFeatures = loadData("./classifier_feature_record.txt")
    # imgNames, imgPreds, imgValues, imgFeatures = loadData("/Users/cxliu/Documents/Code/CS235/deep_facial_feature_comparison/kmeans/classifier_feature_record.txt")

    # imgFeatures = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
    # imgFeatures = np.array(imgFeatures, dtype=np.float64)
    start = time.time()

    # print("--------Kmeans with varianceSelection--------\n")
    # selectedFeatures = varianceSelection(imgFeatures, imgPreds)
    # trainX = imgFeatures[:, selectedFeatures]
    # for i in range(0, 5):
    #     centers, clusters, preds= k_means(trainX, imgPreds, 2, i)

    # print("--------Kmeans with pearsonCorrelationSelection--------\n")
    # selectedFeatures = pearsonCorrelationSelection(imgFeatures, imgPreds)
    # trainX = imgFeatures[:, selectedFeatures]
    # for i in range(0, 5):
    #     centers, clusters, preds= k_means(trainX, imgPreds, 2, i)

    # print("--------Kmeans with fisherScoreSelection--------\n")
    # selectedFeatures = fisherScoreSelection(imgFeatures, imgPreds)
    # trainX = imgFeatures[:, selectedFeatures]
    # for i in range(0, 5):
    #     centers, clusters, preds= k_means(trainX, imgPreds, 2, i)

    # print("--------Kmeans with forwardSelection--------\n")
    # selectedFeatures = forwardSelection(imgFeatures, imgPreds)
    # trainX = imgFeatures[:, selectedFeatures]
    # for i in range(0, 5):
    #     centers, clusters, preds= k_means(trainX, imgPreds, 2, i)

    print("--------Kmeans with backwardSelection--------\n")
    selectedFeatures = backwardSelection(imgFeatures, imgPreds)
    trainX = imgFeatures[:, selectedFeatures]
    for i in range(0, 5):
        centers, clusters, preds= k_means(trainX, imgPreds, 2, i)
    
    cnt = 0
    for i in range(len(preds)):
        if(preds[i] == imgPreds[i]):
            cnt += 1
    # cnt = (np.array(preds, dtype=np.int8) != imgPreds).mean()
    end = time.time()
    print("Consumed time:" + str(end - start) + " Dis cnt:" + str(cnt))
    print('The End')
