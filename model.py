import train_functions as tf
from kernels import kernel_list
import numpy as np
from beta_train import beta_train
from draw import draw_pred_difference
from sklearn.kernel_ridge import KernelRidge  

class MTMK:
    # 初始化
    # 调整Kernel直接调整kernel_list，这里不放kernel的参数的接口了
    def __init__(self,kernel_list:list=kernel_list(),task_num=4) -> None:
        # 任务数量
        self.task_num = task_num
        
        self.kernel_list = kernel_list
        # self.alpha = alpha
        # self.lmd = lmd  # lambda是正则化的参数

        # 需要拟合的参数
        self.eta = []
        self.beta = []

        # 参数的Loss值
        self.loss_eta = []
        self.loss_beta = []

        # 训练集的数据
        self.train_data = []
        pass
    
    
    def fit(self,X_train,Y_train,t0,mu,alpha,lmd,tol,epoch_num) -> None:

        self.train_data = X_train
        
        # 训练eta的代码
        # self.eta, self.loss_eta, AK = tf.eta_train(
        #     KernelFunctionList=self.kernel_list,
        #     DataX=X_train,DataY=Y_train,
        #     t0=t0,
        #     mu=mu,
        #     lmd=lmd,
        #     tol=tol,
        #     alpha=alpha,
        #     Epoch_num=epoch_num)

        # 强行把每个task的eta设置为[1,0,0,0]，测试beta的功能
        for i in range(0,self.task_num):
            tmp = np.zeros(shape=(4,1),dtype=float)
            tmp[0] = 1
            self.eta.append(tmp)
        
        print("MTMK eta training finished")

        self.beta, self.loss_beta = beta_train(
            KernelFunctionList=self.kernel_list,
            DataX=X_train,DataY=Y_train,
            eta=self.eta,alpha=alpha,lmd=lmd,Epoch_num=epoch_num)
        # print(self.beta)
        print("MTMK beta training finished")

        pass

    # 预测，给出预测的结果，连续值或者离散的0/1形式的label    
    # 给出某个任务的预测：
    def predict_single_task(self,task_order:int,X_test_k,continuous:bool=True)->list: 
        # task_order = k
        k = task_order
        # eta_k = self.eta[k]
        beta_k = self.beta[k]
        yk_pred_list = []

        x_train_k = self.train_data[k]
        N_t = x_train_k.shape[0]
        # 遍历samples
        for i in range(X_test_k.shape[0]):
            x = X_test_k[i]
            ret = 0
            # 遍历样本
            for j in range(N_t):
                xj = x_train_k[j]
                ret += beta_k[j]*tf.kernel_combined(
                    self.kernel_list,self.eta,k,x,xj)
            # for j in range(N_t):
            y = 1/(1+np.exp(-ret))

            # 是否离散/连续化预测：
            if continuous: yk_pred_list.append(y)
            else: 
                if y > 0.5 : yk_pred_list.append(1)
                else: yk_pred_list.append(0)
            
        return np.array(yk_pred_list).reshape(-1,1)
    

    def predict_all_task(self,X_test:list,continuous:bool=True) ->list:
        ret = []
        # for x_test_k in X_test:
        for i in range(0,len(X_test)):
            ret.append(self.predict_single_task(
                task_order=i, X_test_k=X_test[i],
                continuous=continuous
            ))
        return ret

# 调试MTMK
if __name__ == "__main__":
    X_data = []
    Y_data = []
    SAMPLES = 100
    # 选择了100个samples，1个task，1维的数据，效果很差
    X_data.append(np.random.uniform(low=0,high=1,size=SAMPLES).reshape(-1,1))
    Y_data.append(1/(1+np.exp(-X_data[0])))
    
    # 很关键一步，将函数变成列表类型
    kernels = kernel_list()
    model = MTMK(kernel_list=kernels,task_num=1)
    model.fit(X_data,Y_data,t0=1,mu=1.1,alpha=0.001,lmd=1,tol=0.0001,epoch_num=5000)
    y_pred = model.predict_all_task(X_data,True)

    model = KernelRidge(alpha=1,kernel="linear")
    model.fit(X_data[0],Y_data[0])
    y_pred_krr = model.predict(X_data[0].reshape(-1,1))

    # print(f"predicted label:{y_pred}")
    # print(f"real label:{Y_data}")

    # 对比随机数据集里，KRR，MTMK，真实数据集的不同
    draw_pred_difference(n_samples=SAMPLES,y_real=Y_data[0],
    y_mtmk=y_pred[0],y_krr=y_pred_krr)
    pass