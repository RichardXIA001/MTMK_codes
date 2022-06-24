import numpy as np
import kernels
import matplotlib.pyplot as plt

class simulation:
    def __init__(self, N_samples, N_tasks,N_dimensions, N_kernels=4) :
        # 生成的样本数据
        self.X_data = [] 
        # 生成的样本标签   
        self.Y_data = []
        # 任务数量
        self.N_TASKS = N_tasks
        # 记录每个任务里面的样本数量
        self.N_SAMPLES = N_samples
        self.N_KERNELS = N_kernels
        self.N_DIMENSIONS = N_dimensions
        self.N_PARAMETERS = 1

        self.Kernel_list = kernels.kernel_list()
        self.beta = []
        self.eta = []    
        self.X_i = []             #用于训练的x_i产生true function的x[i]

    # 产生用于true function的参数
    def get_TF_Xi(self):
        for i in range(0, self.N_TASKS):
            # self.X_i.append(np.linspace(0.5,1,self.N_PARAMETERS).reshape(-1,1))
            self.X_i.append(np.array(1).reshape(-1,1))
    # 产生仿真数据x_data
    def get_uniform(self):
        for i in range(0,self.N_TASKS):
            z = np.zeros((self.N_DIMENSIONS,1),dtype=float)
            one = np.ones((self.N_DIMENSIONS,1),dtype=float)

            x = np.linspace(z,one,self.N_SAMPLES)
            x = x.reshape(self.N_SAMPLES,self.N_DIMENSIONS)
            self.X_data.append(x)
            # np.random.seed(i)
            # self.X_data.append(np.random.uniform(0,1,(self.N_SAMPLES,self.N_DIMENSIONS)))
            
    
    # 产生函数参数eta
    def get_eta(self):
        # 先假定eta在2的倍数的维度，取值相同
        # hyperETA = np.random.uniform(0,1,[1,n_kernels/2])
        E = []
        for t in range(0, self.N_TASKS):
            # # E.append(0.25* np.ones((self.N_KERNELS,1),dtype=float))
            # E.append(np.zeros((self.N_KERNELS,1),dtype=float))
            # s = t - 1
            # if s > -1 and s < self.N_KERNELS :
            #     E[t][s] = 1
            # else :
            #     E[t][0] = 1
            # # self.eta.append(np.array([0.5,0.5,0.5,0.5]))
            tmp = np.zeros(shape=(4,1),dtype=float)
            tmp[0] = 1
            self.eta.append(tmp)
            print(self.eta)
            # self.eta.append(np.array([1.0,0.0,0.0,0],dtype=float))
        self.eta = E
        print(self.eta)
            

    # 生成beta
    def get_beta(self):
        # beta = np.zeros((n_tasks,Nt_samples))
        # self.beta = []
        # bound = 0.1 + 100/self.N_SAMPLES
        bound = 0.2

        for k in range(0, self.N_TASKS):
            # if k % 2 == 0:
            #     bound = -10/(k+1)
            #     b = np.linspace(-bound,bound,self.N_PARAMETERS).reshape(-1,1)
            # else:
            #     bound = 10/(k+1)
            #     b = np.linspace(-bound,bound,self.N_PARAMETERS).reshape(-1,1)
            np.random.seed(2*k)
            b = np.random.uniform(-bound,bound,(self.N_PARAMETERS,1))
            # 人为给定相似性，各个task的beta保证3的倍数的部分均是0
            for i in range(0, self.N_PARAMETERS):
                if i % 4 == 0:
                    b[i] = 0
            
            # b = b* 20/self.N_SAMPLES

            self.beta.append(b)
        print("real beta:")
        print(self.beta)
    
    # 真实函数计算
    def calculator_y(self, x, task_order):
        ind = 0
        # b = self.beta[task_order].reshape(-1,1)
        b = self.beta[task_order]
        # e = self.eta[task_order].reshape(-1,1)
        print(f"task_order:{task_order}")
        e = self.eta[task_order]
        # X = self.X_data[task_order].reshape(-1,1)
        X = self.X_i[task_order]
        # print(X)
        # print(b.shape)
        for i in range(0,self.N_PARAMETERS):
            ker = 0
            for j in range(0,self.N_KERNELS):
                # print(e[j].shape)
                # he = self.Kernel_list[j](x,self.X_data[i])
                # print(he.shape)
                ker += e[j] * self.Kernel_list[j](x,X[i])
            ind += b[i]* ker
        return 1/(1+np.exp(-ind))
    
    # 生成数据标签
    def get_label(self):
        for t in range(0, self.N_TASKS):
            # X = self.X_data[t].reshape(-1,1)
            X = self.X_data[t]
            y = np.zeros((self.N_SAMPLES,1),dtype=float)
            for m in range(0, self.N_SAMPLES):
                y[m] = self.calculator_y(X[m],task_order=t)
                # print(a)
                # if a >= bar:
                #     y[m] = 0
                #     l += 1
            #------------
            # 设定阈值bar

            bar = np.median(y)
            # bar = 0.5
            # print(y)
            # print(bar)
            sum = 0
            for m in range(0,self.N_SAMPLES):
                if y[m] >= bar:
                    y[m] = 1
                    sum += 1
                else:
                    y[m] = 0
            print("task %d N/P ratio:"%t)
            y = y.T
            print(sum/self.N_SAMPLES)
            self.Y_data.append(y)
            # print(y)
            # print("Task %d end"%t)
            # print(l)
        
    # 生成仿真数据
    def generate_samples(self):
        self.get_uniform()
        self.get_beta()
        self.get_eta()
        self.get_TF_Xi()
        self.get_label()
        # print(self.X_data[0].shape)
        # print(self.Y_data[0].shape)
        # print(self.beta[0].shape)
        # print(self.eta[0].shape)
    
    # 进行拟合函数的计算
    def fitted_cal_y(self,R_beta,R_eta,x,task_order=0):
        ind = 0
        # b = self.beta[task_order].reshape(-1,1)
        b = R_beta[task_order]
        # e = self.eta[task_order].reshape(-1,1)
        e = R_eta[task_order]
        # X = self.X_data[task_order].reshape(-1,1)
        X = self.X_data[task_order]
        # print(X)
        # print(b.shape)
        for i in range(0,self.N_SAMPLES):
            ker = 0
            for j in range(0,self.N_KERNELS):
                # print(e[j].shape)
                # he = self.Kernel_list[j](x,self.X_data[i])
                # print(he.shape)
                ker += e[j] * self.Kernel_list[j](x,X[i])
            ind += b[i]* ker
        return 1/(1+np.exp(-ind))

    # ----------------------------------------------------
    # 下方是结果展示部分的代码
    def compare_functions(self,parameters,R_beta,R_eta):
        # parameters = 1000
        x = np.linspace(0,1,parameters).reshape(-1,1)
        y_fitted = []
        y_true = []
        for t in range(0,self.N_TASKS):
            # x = self.X_data[t]
            y1 = np.zeros((parameters,1))
            y2 = np.zeros((parameters,1))
            for i in range(0,parameters):
                # 拟合
                y1[i] = self.fitted_cal_y(R_beta,R_eta,x[i],task_order=t)
                # 真实
                y2[i] = self.calculator_y(x[i],task_order=t)
            y_fitted.append(y1)
            y_true.append(y2)
        return y_fitted, y_true

    # 根据真实的eta和beta画图，目前只支持一维
    def draw_plot(self, R_beta, R_eta):
        para = 1000
        X = np.linspace(0,1,para)
        Y = []
        R_Y = []
        Y,R_Y = self.compare_functions(para,R_beta,R_eta)
        for plt_index in range(1,7):
            plt.subplot(3,2,plt_index)
            
            # for i in range(self.N_SAMPLES):
            #     # 就取第一个任务的数据进行计算y
            #     y = self.calculator_y(X[i],task_order=0)
            #     Y.append(y)
            #     ry = self.fitted_cal_y(R_beta=R_beta,R_eta=R_eta,x=X[i],task_order=plt_index-1)
            #     R_Y.append(ry)
            plt.plot(X,Y[plt_index-1],label='fitted function')
            plt.plot(X,R_Y[plt_index-1],label='true function')
            plt.xlabel("xlabel")
            plt.ylabel("ylabel")
            plt.ylim((0,1))
            # plt.title("Comparison between true function and fitted function")
            plt.title("Task %d"%plt_index)
            plt.legend()
        plt.show()

    # 计算MSE:
    def calculate_MSE(self, R_beta, R_eta):
        MSE = []
        parameters = 10 
        x = np.linspace(0,1,parameters).reshape(-1,1)
        # for t in range(0,self.N_TASKS):
        #     sum = 0
        #     # x = self.X_data[t]
        #     for i in range(0,parameters):
        #         # 拟合
        #         y1 = self.fitted_cal_y(R_beta,R_eta,x[i],task_order=t)
        #         # 真实
        #         y2 = self.calculator_y(x[i],task_order=t)

if __name__ == '__main__':
    pass