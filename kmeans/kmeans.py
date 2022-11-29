import numpy as np
import random 
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
        # if(i > 1000):
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

def distance(data, center):
    temp = data - center / 2
    eucDises = np.sqrt(np.sum(np.square(data - center), axis=1))
    return eucDises

def distance2(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))

def k_means(data, k, max_iter=300):
    cs = []
    n = data.shape[0]
    for e in enumerate(random.sample(range(n), k)):
        cs.append(data[e[1]])

    for i in range(max_iter):
        print("开始第{}次迭代".format(i+1))
        preds = []
        clusters = {}
        for j in range(k):
            clusters[j] = []

        eucDisesList = []
        for j in range(k):
            # eucDises = distance(data, data[0])
            eucDises = distance(data, cs[j])
            eucDisesList.append(eucDises)

        for j in range(n):
            minDis = eucDisesList[0][j]
            clusterIdx = 0
            m = 1
            while m < k:
                dis = eucDisesList[m][j]
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
            dis2 = distance2(precs[i], cs[i])
            print("Center " +str(i) + " dis2:" + str(dis2))
            if dis2 > 1e-10:
                conv = False
                break
        if conv == True:  
            break
    return cs, clusters, preds

if __name__ == '__main__':
    imgNames, imgPreds, imgValues, imgFeatures = loadData("classifier_feature_record.txt")

    # imgFeatures = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
    # imgFeatures = np.array(imgFeatures, dtype=np.float64)
    start = time.time()
    centers, clusters, preds= k_means(imgFeatures, 2)
    cnt = 0
    for i in range(len(preds)):
        if(preds[i] == imgPreds[i]):
            cnt += 1
    # cnt = (np.array(preds, dtype=np.int8) != imgPreds).mean()
    end = time.time()
    print("Consumed time:" + str(end - start) + " Dis cnt:" + str(cnt))
    print('The End')
