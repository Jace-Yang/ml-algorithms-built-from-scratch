# !!!  请先阅读 代码文档.pdf 中模型的建立、求解和逐步讲解。

# 作者：吴宇翀 经济统计学 2017310836 https://wuyuchong.com

# 代码开源托管在 https://github.com/wuyuchong/DataMining/tree/master/HomeWork



# ----------------- 导入基本模块 -----------------
import numpy as np
import math
from matplotlib import pyplot as plt

# ------------ 导入source中定义的函数 ------------
from source.sigmoid import sigmoid
from source.sigmoidDerivative import sigmoidDerivative
from source.mse import mse

# ---------------- 定义神经网络类 ----------------
class NeuralNetwork:
    def __init__(self, X, y, learn_rate = 0.1, epochs = 100):
        # 学习率和迭代次数
        self.learn_rate = learn_rate
        self.epochs = epochs
        # 数据
        self.X = X
        self.y = y
        # 权重和截距的初始化
        self.weight_1a = np.random.normal()
        self.weight_1b = np.random.normal()
        self.weight_2a = np.random.normal()
        self.weight_2b = np.random.normal()
        self.weight_o1 = np.random.normal()
        self.weight_o2 = np.random.normal()
        self.bias_1 = np.random.normal()
        self.bias_2 = np.random.normal()
        self.bias_o = np.random.normal()
        # 记录损失函数值
        self.record = np.array([None, None])
    
    # 用于画 MSE 的图像
    def figure(self):
        x = self.record[1:,0]
        y = self.record[1:,1]
        plt.title("Variation of the Loss Function") 
        plt.xlabel("epochs") 
        plt.ylabel("MSE") 
        plt.plot(x,y,"ob") 
        plt.show()
    
    # 定义前馈
    def feedforward(self, x):
        h1 = sigmoid(self.weight_1a * x[0] + self.weight_1b * x[1] + self.bias_1)
        h2 = sigmoid(self.weight_2a * x[0] + self.weight_2b * x[1] + self.bias_2)
        o1 = sigmoid(self.weight_o1 * h1 + self.weight_o2 * h2 + self.bias_o)
        return o1
    
    # 定义预测函数
    def predict(self):
        y_preds = np.apply_along_axis(self.feedforward, 1, self.X)
        return y_preds 
    
    # 定义训练函数
    def train(self):
        for epoch in range(self.epochs):
            for x, y_true in zip(self.X, self.y):
                # 初次前馈
                sum_h1 = self.weight_1a * x[0] + self.weight_1b * x[1] + self.bias_1
                h1 = sigmoid(sum_h1)
        
                sum_h2 = self.weight_2a * x[0] + self.weight_2b * x[1] + self.bias_2
                h2 = sigmoid(sum_h2)
        
                sum_o1 = self.weight_o1 * h1 + self.weight_o2 * h2 + self.bias_o
                o1 = sigmoid(sum_o1)
                y_pred = o1
        
                # 计算导数
                L_ypred = -2 * (y_true - y_pred)
        
                # 输出层
                ypred_weight_o1 = h1 * sigmoidDerivative(sum_o1)
                ypred_weight_o2 = h2 * sigmoidDerivative(sum_o1)
                ypred_bias_o = sigmoidDerivative(sum_o1)
        
                ypred_h1 = self.weight_o1 * sigmoidDerivative(sum_o1)
                ypred_h2 = self.weight_o2 * sigmoidDerivative(sum_o1)
        
                # 隐藏层 1
                h1_weight_1a = x[0] * sigmoidDerivative(sum_h1)
                h1_weight_1b = x[1] * sigmoidDerivative(sum_h1)
                h1_bias_1 = sigmoidDerivative(sum_h1)
        
                # 隐藏层 2
                h2_weight_2a = x[0] * sigmoidDerivative(sum_h2)
                h2_weight_2b = x[1] * sigmoidDerivative(sum_h2)
                h2_bias_2 = sigmoidDerivative(sum_h2)
        
                # # 迭代权重和偏差
                # 隐藏层 1
                self.weight_1a -= self.learn_rate * L_ypred * ypred_h1 * h1_weight_1a
                self.weight_1b -= self.learn_rate * L_ypred * ypred_h1 * h1_weight_1b
                self.bias_1 -= self.learn_rate * L_ypred * ypred_h1 * h1_bias_1
        
                # 隐藏层 2
                self.weight_2a -= self.learn_rate * L_ypred * ypred_h2 * h2_weight_2a
                self.weight_2b -= self.learn_rate * L_ypred * ypred_h2 * h2_weight_2b
                self.bias_2 -= self.learn_rate * L_ypred * ypred_h2 * h2_bias_2
        
                # 输出层
                self.weight_o1 -= self.learn_rate * L_ypred * ypred_weight_o1
                self.weight_o2 -= self.learn_rate * L_ypred * ypred_weight_o2
                self.bias_o -= self.learn_rate * L_ypred * ypred_bias_o
    
            # 计算损失函数，并保存
            y_preds = np.apply_along_axis(self.feedforward, 1, self.X)
            loss = mse(self.y, y_preds)
            new = np.array([epoch, loss])
            self.record = np.vstack([self.record, new])
