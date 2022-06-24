# 这个文件只用来画图
import matplotlib.pyplot as plt
import numpy as np

def draw_pred_difference(n_samples,y_real,y_mtmk,y_krr):
    x = np.arange(n_samples)
    plt.plot(x,y_real)
    plt.plot(x,y_mtmk)
    plt.plot(x,y_krr)
    plt.legend(['real label','our model','KRR'])
    plt.show()
