import numpy as np
import matplotlib.pyplot as plt
def obj_fun(x): return x*x
def dir_fun(x): return 2*x
x_list = []
y_list = []
def minimize(init_x, lr = 0.1, dif = 1e-9, max_iter = 1000):
    x0 = init_x
    y0 = obj_fun(x0)
    x_list.append(x0)
    y_list.append(y0)
    for i in range(max_iter):
        x1 = x0 - lr*dir_fun(x0)
        y1 = obj_fun(x1)
        x_list.append(x1)
        y_list.append(y1)
        if abs(y1-y0) <= dif:
            print("是否收敛： True", "极小点：(%e, %e)" % (x1, y1), "循环次数: %d" % i )
            return
        x0 = x1
        y0 = y1
    print("是否收敛: False", "极小点: NaN", "循环次数: %d" % max_iter)
minimize(2.0)
plt.plot(x_list, y_list, "o")
plt.show()