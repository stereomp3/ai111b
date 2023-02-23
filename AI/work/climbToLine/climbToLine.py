import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

from regex import P

# x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
# y = np.array([2, 3, 4, 5, 6], dtype=np.float32)
x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)


def predict(a, xt):
    return a[0] + a[1] * xt


def MSE(a, x, y):
    total = 0
    for i in range(len(x)):
        total += (y[i] - predict(a, x[i])) ** 2
    return total


def loss(p):
    return MSE(p, x, y)


def ArrayAdd(l1, l2, sub=False):
    l1c = l1.copy()
    for i in range(len(l1c)):
        if not sub:
            l1c[i] += l2[i]
        else:
            l1c[i] -= l2[i]
    return l1c


print(loss([2, 1]))


# p = [0.0, 0.0]
# plearn = optimize(loss, p, max_loops=3000, dump_period=1)
def hillClimbing(f, hx, h=0.01):
    h_x = hx.copy()
    h1 = [h, 0]
    h2 = [0, h]
    print(f(h_x + h1))
    while True:
        print(h_x)
        print(f(h_x))
        In = False
        if f(ArrayAdd(h_x, h1)) < f(h_x):
            h_x = ArrayAdd(h_x, h1)
            In = True
        if f(ArrayAdd(h_x, h1, True)) < f(h_x):
            h_x = ArrayAdd(h_x, h1)
            In = True
        if f(ArrayAdd(h_x, h2)) < f(h_x):
            h_x = ArrayAdd(h_x, h2)
            In = True
        if f(ArrayAdd(h_x, h2, True)) < f(h_x):
            h_x = ArrayAdd(h_x, h2, True)
            In = True
        if not In:
            break
    return h_x


def optimize():
    # 請修改這個函數，自動找出讓 loss 最小的 p
    # p = [2,1] # 這個值目前是手動填的，請改為自動尋找。(即使改了 x,y 仍然能找到最適合的回歸線)
    # p = [3,2] # 這個值目前是手動填的，請改為自動尋找。(即使改了 x,y 仍然能找到最適合的回歸線)
    p = [0.0, 0.0]
    return hillClimbing(loss, p)


p = optimize()

# Plot the graph
y_predicted = list(map(lambda t: p[0] + p[1] * t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
