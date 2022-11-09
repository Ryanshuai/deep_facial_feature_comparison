import numpy as np
from sklearn.metrics import  confusion_matrix

with open("classifier_kmeans_result.txt", "r") as f:
    lines = f.readlines()

line = lines[0]
y_pred = np.array(list(map(int, line.strip().split(" "))))

with open("classifier_feature_record.txt", "r") as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
y_label = [int(line.split("  ")[1]) for line in lines]

print("Confusion Matrix: \n", confusion_matrix(y_label, y_pred))

