import numpy as np
import matplotlib.pyplot as plt

def p(x):
    mu1 = 3
    mu2 = 10
    v1 = 10
    v2 = 3
    return 0.3*np.exp(-(x-mu1)**2/v1) + 0.7*np.exp(-(x-mu2)**2/v2)


def q(x, mu, sigma):
    return np.exp(-(x-mu)**2/(sigma**2))

N = 5000
y = np.zeros(N)
mu =5
sigma= 10
y[0] = np.random.normal(mu, sigma)  # 根据q产生一个样本
for i in range(N-1):
    ynew = np.random.normal(mu,sigma)  # 从正态分布产生一个样本
    alpha = min(1, p(ynew)*q(y[i], mu, sigma)/(p(y[i])*q(ynew, mu, sigma)))  # 计算接受概率
    if np.random.rand() < alpha:
        y[i+1] = ynew
    else:
        y[i+1] = y[i]

# # random walk chain
# sigma = 10
# u2 = np.random.rand(N)
# y2 = np.zeros(N)
# y2[0] = np.random.normal(0,sigma)
# for i in range(N-1):
#     y2new = y2[i] + np.random.normal(0,sigma)
#     alpha = min(1,p(y2new)/p(y2[i]))
#     if u2[i] < alpha:
#         y2[i+1] = y2new
#     else:
#         y2[i+1] = y2[i]

plt.figure(1)
plt.hist(y, density=1)
x = np.arange(0,4,0.01)
plt.plot(x, p(x))
# plt.figure(2)
# nbins = 30
# plt.hist(y2, bins = x)
# plt.plot(x, px*N/sum(px), color='r', linewidth=2)

plt.show()