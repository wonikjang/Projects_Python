import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Number of classes
N = 10
mean_class = 0
std = 2

x = np.linspace(0, N, N, endpoint=False)

y = multivariate_normal.pdf(x, mean = mean_class , cov = std);
y = y / sum(y);

mean_class1 = 9
std1 = 2

y1 = multivariate_normal.pdf(x, mean = mean_class1 , cov = std1);
y1 = y1 / sum(y1);

plt.xticks(x)
plt.plot(x, y, color='b')
plt.plot(x, y1, color='r')

plt.title("Gaussian-PDF")
plt.xlabel("index" )
plt.ylabel("Gaussian PDF" )


plt.tight_layout()

plt.show()




