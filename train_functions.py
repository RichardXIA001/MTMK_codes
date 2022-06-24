"""
functions about training
"""

import numpy as np
from scipy.optimize import minimize

def alignment(KernelFunction, DataX, DataY):
    """
    Return the alignment of this kernel on data. Value 0~1

    KernelFunction: kernel function of two embedding input
    DataX: N x P matrix, each row represents a sample (one embedding pair)
    DataY: N x 1 vector
    """
    n = DataX.shape[0]    # number of samples
    
    Matrix_Y = DataY.reshape((n,1)).dot(DataY.reshape((1,n))) # ideal kernel
    Matrix_kernel = np.ones((n,n))
    for i in range(n):
        xi = DataX[i]
        for j in range(n):
            xj = DataX[j]
            Matrix_kernel[i][j] = KernelFunction(xi,xj)

    inner_product_K = Matrix_kernel*Matrix_kernel
    inner_product_K_Y = Matrix_kernel*Matrix_Y

    alignment_kernel = inner_product_K_Y.sum()/(n*(inner_product_K.sum())**0.5)

    return alignment_kernel
    

def alignment_matrix(KernelFunctionList,DataX, DataY):
    """
    Return alignments matrix of different kernels on different tasks
    form: T X P matrix

    KernelFunctionList: list of kernel functions (# = P)
    DataX: list of different tasks' x data 
        (# = T, each x's shape = (N_t, x-dimesions))
    DataY: list of different tasks' y data 
        (# = T, each y's shape = (N_t, 1))
    """
    
    P = len(KernelFunctionList)
    T = len(DataX)
    Matrix_alignment = np.ones((T,P))
    
    for t in range(T):
        DataXt = DataX[t]
        DataYt = DataY[t]
        sum_alignment = 0
        for p in range(P):
            kernelfunction = KernelFunctionList[p]
            Matrix_alignment[t][p] = alignment(KernelFunction=kernelfunction,DataX=DataXt,DataY=DataYt)
            sum_alignment += Matrix_alignment[t][p]
        # normalize
        for p in range(P):
            Matrix_alignment[t][p] = Matrix_alignment[t][p]/sum_alignment

    # print("Matrix_alignment")
    # print(Matrix_alignment)
    return Matrix_alignment

# 没屁用的函数
def loss_eta(eta, Matrix_alignment, lmd):
    """
    Return the loss of loss_eta

    eta: T X P matrix
    Matrix_alignment: T X P matrix
    """
    T = eta.shape[0]
    P = eta.shape[1]

    loss_matrix_1 = abs(eta) * np.log(Matrix_alignment)
    # print("Matrix Alignment")
    # print(Matrix_alignment)
    # loss_matrix_2 = (1-eta) *np.log(np.ones((T,P))- Matrix_alignment)
    loss_2 = lmd*sum(abs(eta).max(axis=0))
    # return -1*(loss_matrix_1.sum() + loss_matrix_2.sum() )+ loss_2
    # print("test 1:")
    # print(-1*loss_matrix_1.sum())
    return -1*loss_matrix_1.sum() + loss_2


def eta_u_train(KernelFunctionList,DataX, DataY, eta_0, u_0, Matrix_Alignment, t, lmd, alpha, Epoch_num):
    """
    Return the optimal eta and u given parameter t

    eta_0: T X P matrix
    u_0: P X 1 matrix
    Matrix_Alignment: T X P matrix
    t: parameter in Loss Barrier
    lmd: highparameter
    alpha: 
    Epoch_num: 
    """

    T = len(DataX)  # task number
    P = len(KernelFunctionList) # kernel number

    # initialize the partial derivative and eta and u 
    p_eta = eta_0
    p_u = u_0
    eta = eta_0
    u = u_0
    
    # newton iteration
    for epoch in range(Epoch_num):
        # intermediate matrix: 1/(u_j**2-eta_ij**2)
        inter_matrix = np.ones((T,P))
        for i in range(T):
            for j in range(P):
                inter_matrix[i][j] = 1/(u[j][0]**2 - eta[i][j]**2)

        # calculate the partial derivative
        # np.log(np.ones((T,P))-Matrix_Alignment) 
        p_eta = -np.sign(eta)*np.log(Matrix_Alignment) + np.log(np.ones((T,P))-Matrix_Alignment) + 2/t * eta * inter_matrix
        p_u = lmd + (-2/t) * np.sum(inter_matrix.T * u, axis=1).reshape((P,1))

        eta = eta - alpha * p_eta
        u = u - alpha * p_u

    return eta, u

def loss_barrier(eta,u,Matrix_alignment,lmd,t):
    """
    coef: array form of [eta(T x P), u(P x 1)]
    """
    # eta = coef[0]
    # u = coef[1]
    T = eta.shape[0]
    P = eta.shape[1]
    # cross entropy
    # loss_matrix_1 = -eta*np.log(Matrix_alignment)
    # print("MatrixAlignment")
    # print(Matrix_alignment)

    loss_matrix_1 = -np.sqrt(eta)*np.log(Matrix_alignment)
    loss_1 = loss_matrix_1.sum()
    # penalty
    loss_2 = lmd*u.sum()
    # barrier
    loss_matrix_2 = eta.copy()
    for i in range(T):
        loss_matrix_2[i,] = u.flatten() + loss_matrix_2[i,]
    loss_matrix_2 = np.log(loss_matrix_2)

    loss_matrix_3 = eta.copy()
    # print(eta)
    # print(u)
    for i in range(T):
        loss_matrix_3[i,] = u.flatten() - loss_matrix_3[i,]
    loss_matrix_3 = np.log(loss_matrix_3)

    loss_3 = -1/t * (loss_matrix_2.sum()+loss_matrix_3.sum())
    # print(loss_1,loss_2,loss_matrix_2.sum(),loss_matrix_3.sum())
    loss = loss_1 + loss_2 + loss_3
    # print("loss 1 : %.2f, Loss 2: %.2f, loss 3: %.2f"%(loss_1,loss_2,loss_3))
    return loss


def eta_u_train_optimize(KernelFunctionList, eta_0, u_0, Matrix_Alignment, t, lmd):
    T = eta_0.shape[0]
    P = eta_0.shape[1]
    def fun(coef):
        eta = coef[0:T*P].reshape((T,P))
        u = coef[T*P:T*P+P].reshape((P,1))
        return loss_barrier(eta,u, Matrix_Alignment, lmd, t)
    
    def cons_eq(coef):
        "sum of eta in each task = 1"
        eta = coef[0:T*P].reshape((T,P))
        # u = coef[T*P:T*P+P].reshape((P,1))
        return eta.sum(axis=1)-np.ones(T)

    def cons_ineq(coef):
        "eta >= 0"
        # eta = coef[0:T*P].reshape((T,P))
        # u = coef[T*P:T*P+P].reshape((P,1))
        # eta = coef[0]
        # T = eta.shape[0]
        # P = eta.shape[1]
        return coef[0:T*P] - 1e-5 * np.ones(T*P)

    cons = (
        {'type':'eq','fun':cons_eq},
        {'type':'ineq','fun':cons_ineq}
    )
    bnds = tuple([(0,1) for _ in range(T*P+P)])

    coef_0 = np.append(eta_0.flatten(), u_0.flatten())
    # print(coef_0.shape)
    # fun(coef_0)
    # res = minimize(fun, coef_0,method='SLSQP',constraints=cons,bounds=bnds,options ={"maxiter":500})
    
    res = minimize(fun, coef_0,method='SLSQP',constraints=cons)
    # method="SLSQP",
    loss_result = res.fun
    coef_result = res.x
    
    eta = coef_result[0:T*P].reshape((T,P))
    u = coef_result[T*P:T*P+P].reshape((P,1))
    # print('迭代终止是否成功：', res.success)
    # print('迭代终止原因：', res.message)
    return loss_result, eta, u

def eta_train(KernelFunctionList,DataX, DataY, t0, mu, lmd, tol ,alpha, Epoch_num):
    """
    Return the optimal eta of eta problem. T X P matrix.
    
    KernelFunctionList: list of kernel functions (# = P)
    DataX: list of different tasks' x data 
        (# = T, each x's shape = (N_t, x-dimesions))
    DataY: list of different tasks' y data 
        (# = T, each y's shape = (N_t, 1))
    
    Hyperparameters:
        t0, mu(>1), lmd, tol, alpha(<1), Epoch_num(num of iterations for each t), 
    """
    # Some parameters
    T = len(DataX)  # task number
    P = len(KernelFunctionList) # kernel number
    m = 2*T*P   # parameter for iteration
    t = t0
    # Initialize eta matrix, u vector
    eta_t = 0.1*np.ones((T, P))
    u_t = 0.2*np.ones((P,1))

    # calculate alignment matrix
    Matrix_Alignment = alignment_matrix(KernelFunctionList,DataX, DataY)
    loss_eta_list = []
    while(m/t>tol):
        # print(f"t={t}")
        # solve the optimization problem with Matrix_eta as start
        """
        # Method: Mini-Batch Gradient Descent
        # print(f"t={t},")
        eta_optimal, u_optimal = \
            eta_u_train(KernelFunctionList,DataX, DataY, eta_t, u_t, Matrix_Alignment, t, lmd, alpha, Epoch_num)

        # use the optimal as start, solve next problem with new t
        eta_t = eta_optimal
        u_t = u_optimal

        t = t*mu
        loss_eta_list.append(loss_eta(eta_t, Matrix_Alignment,lmd))
        """
        loss_now, eta_t, u_t = eta_u_train_optimize(KernelFunctionList, eta_t, u_t, Matrix_Alignment, t, lmd)
        loss_eta_list.append(loss_now)
        t = t*mu
    # loss_eta_list: loss function（看下降趋势），Matrix_Alignment：
    return eta_t, loss_eta_list, Matrix_Alignment

def kernel_combined(KernelFunctionList,eta,task,x1,x2):
    """
    Return the combined kernel function result given x1, x2 in task 

    task: 0~T-1
    eta: T X P
    """
    eta_task = eta[task]
    result = 0
    for i in range(len(KernelFunctionList)):
        kernel_i = KernelFunctionList[i]
        result += eta_task[i]*kernel_i(x1,x2)
    
    return result

def Matrix_Kernel_Task(KernelFunctionList,DataXt,eta,task):
    N_t = DataXt.shape[0]
    Matrix_K_T = np.ones((N_t,N_t))
    for i in range(N_t):
        for j in range(N_t):
            xi = DataXt[i]
            xj = DataXt[j]
            Matrix_K_T[i][j] = kernel_combined(KernelFunctionList,eta,task,xi,xj)

    return Matrix_K_T

def loss_beta_task(Matrix_K_t,beta,y,lmd,N_t):

    temp_matrix_1 = y*np.dot(Matrix_K_t,beta)
    M_temp = Matrix_K_t.dot(beta)
    v_temp = beta.T.dot(M_temp)[0][0]
    # print("beta loss")
    resu = lmd/N_t*v_temp + sum(np.log(1+np.exp(-temp_matrix_1)))[0]/N_t
    # print(resu)
    return resu


def beta_train_task(KernelFunctionList,DataXt,DataYt,eta,task, alpha,lmd, Epoch_num):
    """
    Return beta in one task.
    Given eta, we can derive the combined kernel in each task.
    Here we do kernel logistics regression for each task. 

    Matrix_K_t: symmetric matrix, 计算了xi和xj的kernel的线性组合的值
    """
    Matrix_K_t = Matrix_Kernel_Task(KernelFunctionList,DataXt,eta,task)
    # initialize beta
    N_t = DataXt.shape[0]
    coef_beta_0 = np.ones(N_t)
    y = DataYt.reshape((N_t,1))
    """
    loss_t_list = []
    
    for epoch in range(Epoch_num):
        P_beta_1 = 2*Matrix_K_t.dot(beta)*lmd/N_t
        temp_matrix_1 = y*np.dot(Matrix_K_t,beta)
        
        
        for j in range(temp_matrix_1.shape[0]):
            exp_term = temp_matrix_1[j][0]
            if(exp_term>100):
                temp_matrix_1[j][0] = 100
            elif(exp_term<-100):
                temp_matrix_1[j][0] = -100
            # else:
            #     temp_matrix_1[j][0] = np.exp(exp_term)
        temp_matrix = np.exp(temp_matrix_1)

        P_beta_2 = -Matrix_K_t.dot(y/(1 + temp_matrix))/N_t
        P_beta = (P_beta_1 + P_beta_2)
        # if(max(abs(P_beta[:,0]))<0.001):
        #     print("diff<0.001")
        #     break
        beta = beta - alpha*(P_beta_1+P_beta_2)
        M_temp = Matrix_K_t.dot(beta)
        v_temp = beta.T.dot(M_temp)[0][0]
        loss = lmd/N_t*v_temp + sum(np.log(1+np.exp(-temp_matrix_1)))[0]/N_t
        loss_t_list.append(loss)
    return beta, np.array(loss_t_list)
    """
    """use minimize method"""
    def fun(coef):
        beta = coef.reshape((-1,1))
        return loss_beta_task(Matrix_K_t,beta,y,lmd,N_t)

    # 这行干啥的？？？？
    res = minimize(fun, coef_beta_0,method='SLSQP')
    # method="SLSQP",
    loss_result = res.fun
    coef_result = res.x
    beta = coef_result.reshape((N_t,1))

    # print('迭代终止是否成功：', res.success)
    # print('迭代终止原因：', res.message)
    return beta,loss_result

def beta_train(KernelFunctionList,DataX,DataY,eta, alpha,lmd, Epoch_num):
    """
    Return beta_list of all task.
    """
    beta_list = []
    loss_beta = 0.0*np.array(range(Epoch_num))
    for task in range(len(DataX)):
        DataXt = DataX[task]
        DataYt = DataY[task]
        beta_task,loss_beta_t = beta_train_task(KernelFunctionList,DataXt,DataYt,eta,task,alpha,lmd,Epoch_num)
        beta_list.append(beta_task)
        loss_beta += loss_beta_t

    return beta_list, loss_beta


def predict(KernelFunctionList, eta, beta_t,DataXt, TestXt,task):
    """
    Return predicted y given a new x and its task
    """
    y_list = []
    for i in range(TestXt.shape[0]):
        x = TestXt[i]
        N_t = DataXt.shape[0]
        result = 0 
        for i in range(N_t):
            xi = DataXt[i]
            result += beta_t[i][0]*kernel_combined(KernelFunctionList,eta,task,xi,x)
        y = 1/(1+np.exp(-result))
        y_list.append(y)
    
    return np.array(y_list)
    

