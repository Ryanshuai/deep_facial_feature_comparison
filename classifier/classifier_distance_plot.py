import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

with open("classifier_kmeans_center_distance.txt", "r") as f:
    lines = f.readlines()

center = "1"
lines = np.array([float(line.strip().split(" ")[1]) for line in lines if line.startswith(center)])

x = lines
mu = np.mean(x)
std = np.std(x)

plt.xlabel("Distance to Cluster Center_"+center)
plt.ylabel("count")

h = 120
text_color = "white"

plt.axvline(x=mu, color="red")
plt.text(mu, h, 'mean', rotation=90, color=text_color)

plt.axvline(x=mu - std, color="coral")
plt.text(mu - std, h, 'mean-std', rotation=90, color=text_color)

plt.axvline(x=mu + std, color="coral")
plt.text(mu + std, h, 'mean+std', rotation=90, color=text_color)

plt.hist(x, bins=100)
plt.title("Histogram of Distance to Cluster Center")

plt.savefig(f'distance_hist_cluster_{center}.png', dpi=300)
plt.show()
