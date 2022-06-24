# %%
import numpy as np

import matplotlib.pyplot as plt  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge  

from tqdm import tqdm
import importlib as imp
import train_functions
import kernels
import simulation_functions as sf
imp.reload(sf)

from draw import draw_pred_difference as draw_pred
# %%
# 生成仿真数据
# 初始化参数
N_SAMPLES = 10
N_TASKS = 2
N_KERNELS = 4
DIMENSIONS = 5

X_data = []
Y_data = []

# 生成测试数据
simu = sf.simulation(N_SAMPLES,N_TASKS,DIMENSIONS,N_KERNELS)
# simu.generate_samples()
# X_data = simu.X_data
# Y_data = simu.Y_data
X_data.append(np.random.uniform(low=0,high=1,size=N_SAMPLES).reshape(-1,1))

Y_data.append(np.random.randint(low=0,high=2,size=N_SAMPLES).reshape(-1,1))

# %%
# 默认数据已经被处理好了，已经获得样本矩阵X和label矩阵Y
# 五折验证

KernelFunctionList = kernels.kernel_list()
# 任务数量
K = len(X_data) 





    
X_train = X_data
X_test = X_data
Y_train = Y_data
Y_test  = Y_data

# ----------4. multi task multi kernel

# 1. multi-kernel 
# 1.1 train eta
# eta, loss_eta_list, AK = train_functions.eta_train(KernelFunctionList,DataX=X_train, DataY=Y_train, t0=1, mu=1.1, lmd=0.0, tol=0.0001 ,alpha=0.001, Epoch_num=500)
# eta, loss_eta_list, AK = train_functions.eta_train(KernelFunctionList,DataX=X_train, DataY=Y_train, t0=1, mu=1.1, lmd=5.7, tol=0.0001 ,alpha=0.001, Epoch_num=5000)

eta = []
for i in range(0,N_SAMPLES):
    tmp = np.zeros(shape=(4,1),dtype=float)
    tmp[0] = 1
    eta.append(tmp)
    # eta.append(np.array([1,0,0,0]).reshape(-1,1))

# print(loss_eta_list)
print("eta finished")


# using AK to replace eta
# beta, loss_beta_array = train_functions.beta_train(KernelFunctionList,DataX=X_train, DataY=Y_train,eta=AK,alpha=0.005,lmd=1,Epoch_num=500)

# 1.2 train beta
beta, loss_beta_array = train_functions.beta_train(KernelFunctionList,DataX=X_train, DataY=Y_train,eta=eta,alpha=0.01,lmd=5,Epoch_num=5000)
# beta, loss_beta_array = train_functions.beta_train(KernelFunctionList,DataX=X_train, DataY=Y_train,eta=eta,alpha=0.5,lmd=0.951,Epoch_num=10000)
# print(loss_beta_array)
# print(beta)

# a = np.mean(beta,axis=0)
# b = np.std(beta,axis=0)
# print("beta: mean=%.6f, std=%.6f"%(a,b))
# print("range of beta: [%.6f, %.6f]"%(a-3*b,a+3*b))

print("beta finished")
# # 1.3 prediction
# y_predict_multi = []
# for k in range(len(X_test)):
#     beta_t = beta[k]
#     DataXt = X_train[k]
#     TestXt = X_test[k]
#     y_pred_k = train_functions.predict(KernelFunctionList, eta, beta_t=beta_t,DataXt=DataXt, TestXt=TestXt,task=k)
#     y_predict_multi.append(y_pred_k)


print("multi finished")
# save auc result
for k in range(len(X_test)):
    X_test_k = X_test[k]
    Y_test_k = Y_test[k][:,0]
    
    # 1. multitask multikernel prediction
    beta_t = beta[k]
    DataXt = X_train[k]
    TestXt = X_test[k]
    y_pred_k = train_functions.predict(KernelFunctionList, eta, beta_t=beta_t,DataXt=DataXt, TestXt=TestXt,task=k)
    print(y_pred_k.shape)
    # fpr,tpr,threshold = roc_curve(Y_test_k, y_pred_k) ###计算真正率和假正率
    # auc_multi_k = auc(fpr,tpr) ###计算auc的值
    # if auc_multi_k < 1:
    #     pass
    # else:
    #     print("test %d"%k)
    #     print(Y_test_k)
    #     print("pred %d"%k)
    #     print(y_pred_k)

    # MT_MK_list.append(auc_multi_k)


    # multi_result_list.append(0)
# 计算加权平均的auc值，(只计算multi时)
# multi_result_sum = 0

# for k in range(len(X_test)):
#     num = second_column[k]
#     multi_result_sum += num * MT_MK_list[k]
# num_sum = second_column[K]
# # print(multi_result_sum/num_sum)
# MT_MK_list.append(multi_result_sum/num_sum)

# # 存入df_result表格
# df_result[f"ST_SK_{seed}_{fold+1}"] = ST_SK_list
# df_result[f"ST_MK_{seed}_{fold+1}"] = ST_MK_list
print("eta:")
# print(eta)
print("beta")
# print(beta)
# 绘制Loss曲线

# for plt_index in range(1,7):
#     plt.subplot(3,4,plt_index)
#     plt.plot(range(len(loss_eta_list[plt_index-1])),loss_eta_list[plt_index-1],
#     label='eta loss')
#     plt.title("Task %d eta loss"%plt_index)
#     plt.legend()
# for plt_index in range(7,13):
#     plt.subplot(3,4,plt_index)
#     plt.plot(range(len(loss_beta_array[plt_index-1])),loss_beta_array[plt_index-1],
#     label='fitted function')
#     plt.title("Task %d eta loss"%(plt_index-6))
#     plt.legend()
# plt.show()

# simu.draw_plot(beta,eta)
print("finished")
# print(MT_MK_list)

# %%
# 调用sklearn库里面的KRR
krr = KernelRidge(alpha=1,kernel="linear")
# print(X_data[0].shape)
# print(Y_data[0].shape)

krr.fit(X_data[0],Y_data[0].reshape(-1,1))

# print(krr.dual_coef_)
# %%
y_pre = krr.predict(X_data[0].reshape(-1,1))
# print(y_pre)

coe1 = krr.dual_coef_
coe2 = beta
difference = []
pred_diff = []
# for i in range(0, len(beta)):
#     difference.append(coe1[i] - coe2[i])
#     print(difference[i])
#     # pred_diff.append(y_pre[i] - y_pred_k[i])
#     # print(y_pre[i] - y_pred_k[i])
draw_pred(N_SAMPLES,y_pre,y_pred_k)
