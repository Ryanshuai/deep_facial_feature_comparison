import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

with open("vae_mse_record.txt", "r") as f:
    lines = f.readlines()
lines = np.array([float(line.strip()) for line in lines])

x = lines
mu = np.mean(x)
std = np.std(x)

plt.xlabel("MSE")
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
plt.title("Histogram of MSE")

plt.savefig('mse_hist.png', dpi=300)
plt.show()
