import train_functions as tf
import numpy as np
from scipy.optimize import minimize

# 计算单个任务下的beta的Loss
def loss_beta_task(Matrix_K_t,beta,y,lmd,N_t):
    temp_matrix_1 = y*np.dot(Matrix_K_t,beta)
    # print(f"temp size: {temp_matrix_1.shape}")
    M_temp = Matrix_K_t.dot(beta)
    v_temp = beta.T.dot(M_temp)[0][0]
    # print(f"v_temp size: {v_temp.shape}")
    # print("beta loss")
    resu = lmd/N_t*v_temp + sum(np.log(1+np.exp(-temp_matrix_1)))[0]/N_t
    # print(f"loss:{resu}")
    return resu

# 对单个任务进行参数训练
def beta_train_task(KernelFunctionList,DataXt,DataYt,
eta,task, alpha,lmd, Epoch_num):

    Matrix_K_t = tf.Matrix_Kernel_Task(KernelFunctionList,DataXt,eta,task)
    # 单个任务下的samples数量
    N_t = DataXt.shape[0]
    # 单个任务下的beta系数
    coef_beta_0 = np.ones(N_t)
    
    y = DataYt.reshape((N_t,1))

    def fun(coef):
        beta = coef.reshape((-1,1))
        return loss_beta_task(Matrix_K_t,beta,y,lmd,N_t)

    # 用minimize函数进行优化
    res = minimize(fun, coef_beta_0,method='SLSQP')
    # method="SLSQP","BFGS","L-BFGS-B"
    loss_result = res.fun
    coef_result = res.x
    beta = coef_result.reshape((N_t,1))

    # print('迭代终止是否成功：', res.success)
    # print('迭代终止原因：', res.message)
    return beta,loss_result

def beta_train(KernelFunctionList,DataX,DataY,
eta, alpha,lmd, Epoch_num):
    # 记录所有beta
    beta_list = []
    # 任务数量
    # task_num = len(DataX)
    loss_beta = 0.0*np.array(range(Epoch_num))
    for task in range(len(DataX)):
        DataXt = DataX[task]
        DataYt = DataY[task]
        beta_task,loss_beta_t = beta_train_task(KernelFunctionList,DataXt,DataYt,eta,task,alpha,lmd,Epoch_num)
        beta_list.append(beta_task)
        loss_beta += loss_beta_t

    return beta_list, loss_beta
