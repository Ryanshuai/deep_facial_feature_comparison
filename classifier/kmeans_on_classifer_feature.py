from sklearn.cluster import KMeans
import numpy as np
import time

print(time.localtime(time.time()))
with open("classifier_feature_record.txt", "r") as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
features = [line.split("  ")[3] for line in lines]
features = np.array([list(map(float, feature[1:-1].split(","))) for feature in features])

print(time.localtime(time.time()))

kmeans = KMeans(n_clusters=2, max_iter=300).fit(features)

with open("classifier_kmeans_result.txt", "w") as f:
    for label in kmeans.labels_:
        f.write(f"{label} ")

print(kmeans.labels_)
print(kmeans.cluster_centers_)

print(time.localtime(time.time()))
