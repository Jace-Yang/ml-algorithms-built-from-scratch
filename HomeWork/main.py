# Logistic 回归 和 神经网络算法

# 请先阅读 代码文档.pdf 中模型的建立、求解和逐步讲解。

# 此文件中只包含测试用例 Demo

# 作者：吴宇翀 经济统计学 2017310836 https://wuyuchong.com

# 代码开源托管在 https://github.com/wuyuchong/DataMining/tree/master/HomeWork



## ----------------------------------------------------------------------------------------------------
# 使用 sklearn 中的数据集作为测试用例
import numpy as np
import math
from sklearn import datasets



## ----------------------------------------------------------------------------------------------------
# Logistic回归 算法测试用例

import numpy as np
import math
from models.LogisticRegression import LogisticRegression

iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X = X[y!=2]
y = y[y!=2]

# 将学习率固定在 0.01
Logstic = LogisticRegression(X, y, threshold = 0.5)    
Logstic.fit(alpha = 0.01, accuracy = 0.001)
print("epoch:", Logstic.epoch)
print("theta:", Logstic.thetas)
y_predict = Logstic.predict()
y_predict

# 使用自动控制的下降学习率
Logstic2 = LogisticRegression(X, y, threshold = 0.5)    
Logstic2.auto_fit(accuracy = 0.001)
print("epoch:",Logstic2.epoch)
print("theta:",Logstic2.thetas)
y_predict = Logstic2.predict()
y_predict

## ----------------------------------------------------------------------------------------------------
# 神经网络 算法测试用例

from models.NeuralNetwork import NeuralNetwork

iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X = X[y!=2][:,0:2]
y = y[y!=2]

np.random.seed(1)
model = NeuralNetwork(X, y, learn_rate = 0.1, epochs = 1000)    
model.train()
model.predict()
model.figure()

## ----------------------------------------------------------------------------------------------------
# 详解见 代码文档.pdf 
