# ReadMe
## 仿真进度
1. 仿真流程和想法：[overleaf文档](https://www.overleaf.com/project/6045fdaf320b215ed6547e29)

2. 目前的主要问题：

    * 函数拟合效果过差

3. 当前的实验进度（截止到6.24）：

    * 本周只选择了1个kernel，只考虑$\beta$的训练（在`beta_train.py`文件里面）。经过研读学长代码，并没有发现问题（除非python库函数里的`scipy.optimize.minimize`有问题），但是预测的结果和`KRR`以及真实的数据集依然相差较大。

    * 说明：直接运行`model.py`文件即可复现结果

## 代码文件功能说明：
1. simulation文件夹（可以暂时略过）：
    * `simulation.py`：之前用于仿真的文件，点击运行即可（暂时没用）

    * `simulation_funcitons.py`：提供了`simulation`类，可以生成仿真数据集

    * `test.py`：没啥用

2. `kernels.py`：提供了`kernel function`的接口，其中的`kernel_list()`函数可以返回内含4种`kernel`函数对象的列表

3. `model.py`：提供了`MTMK`模型的类，可以进行训练和预测。

4. `beta_train.py`：提供了3个函数，用于$\beta$的训练

5. `draw.py`：提供了画图的函数，用于比较KRR、MTMK的预测结果与真实label的差距

6. `train_functions.py`:学长原来写的用于训练的代码，包含了各种训练参数时可能用到的函数