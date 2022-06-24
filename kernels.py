"""
kernel functions
"""
# %%
import numpy as np

def Gauss1(x1,x2):
    sigma = 20
    return np.exp(-np.linalg.norm(x1-x2)**2 / (sigma**2))

def Gauss2(x1,x2):
    sigma = 10
    # print(np.exp(-np.linalg.norm(x1-x2)**2 / sigma**2))
    return np.exp(-np.linalg.norm(x1-x2)**2 / sigma**2 )

def Gauss3(x1,x2):
    sigma = 1
    return np.exp(-np.linalg.norm(x1-x2)**2 / sigma**2 )

def linear(x1,x2):
    # return sum(x1*x2)/len(x1)
    return sum(x1*x2)

def poly2(x1,x2):
    d=2
    # print(sum(x1*x2)**d)
    return sum(x1*x2)**d

def poly3(x1,x2):
    d = 3
    # return (sum(x1*x2)/len(x1))**d
    return (sum(x1*x2))**d

def exponential1(x1,x2):
    sigma = 20
    return np.exp(-np.linalg.norm(x1-x2)/sigma)

def exponential2(x1,x2):
    sigma = 10
    # print(np.exp(-np.linalg.norm(x1-x2)/sigma))
    return np.exp(-np.linalg.norm(x1-x2)/sigma)

def exponential3(x1,x2):
    sigma = 1
    return np.exp(-np.linalg.norm(x1-x2)/sigma)

def Cauchy1(x1,x2):
    sigma = 10
    # print(1/(1+np.linalg.norm(x1-x2)**2/sigma))
    return 1/(1+np.linalg.norm(x1-x2)**2/sigma)

def Cauchy2(x1,x2):
    sigma = 1
    return 1/(1+np.linalg.norm(x1-x2)**2/sigma)

def kernel_list():
    # return [Gauss1,Gauss2,Gauss3,linear,poly2,poly3,exponential1,exponential2,exponential3,Cauchy1,Cauchy2]
    # return [Gauss1,Gauss2,exponential3,linear]
    # return [Gauss1,exponential3]
    # return [Gauss2,poly2,exponential2,Cauchy1] # four kernels
    
    # 下面这个是之前用的
    return [Gauss3,poly2,exponential3,Cauchy2]
    # return [linear,poly2,exponential3,Cauchy2]
    
    # return [Gauss1,Gauss2,linear]

# %%
# x = np.array
# Gauss1(x1,x2)

# %%
"""
<function Gauss1 at 0x1548C150>
seed=1
d:\科研\multi-task\multi-kernel\train_functions.py:149: RuntimeWarning: invalid value encountered in log
  loss_matrix_3 = np.log(loss_matrix_3)
eta finished
beta finished
multi finished
0.6228020496224379
<function Gauss2 at 0x1548C8A0>
seed=1
eta finished
beta finished
multi finished
0.8405744336569579
<function Gauss3 at 0x13B555D0>
seed=1
eta finished
beta finished
multi finished
0.7898327939590076
<function linear at 0x13B55588>
seed=1
eta finished
beta finished
multi finished
0.8199029126213593
<function poly2 at 0x015C4348>
seed=1
eta finished
d:\科研\multi-task\multi-kernel\train_functions.py:281: RuntimeWarning: overflow encountered in exp
  coef_0 = np.append(eta_0.flatten(), u_0.flatten())
beta finished
multi finished
0.8199029126213593
<function poly3 at 0x015C4C90>
seed=1
eta finished
beta finished
multi finished
0.8174757281553399
<function exponential1 at 0x0099A4B0>
seed=1
eta finished
beta finished
multi finished
0.8029126213592234
<function exponential2 at 0x015C4A08>
seed=1
eta finished
beta finished
multi finished
0.8029126213592234
<function exponential3 at 0x13F5C078>
seed=1
eta finished
beta finished
multi finished
0.7956310679611651
<function Cauchy1 at 0x015C4E88>
seed=1
eta finished
beta finished
multi finished
0.8114077669902913
<function Cauchy2 at 0x015C4B70>
seed=1
eta finished
beta finished
multi finished
0.8029126213592234
"""