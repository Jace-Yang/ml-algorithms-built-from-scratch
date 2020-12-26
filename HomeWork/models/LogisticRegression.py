# !!!  请先阅读 代码文档.pdf 中模型的建立、求解和逐步讲解。

# 作者：吴宇翀 经济统计学 2017310836 https://wuyuchong.com

# 代码开源托管在 https://github.com/wuyuchong/DataMining/tree/master/HomeWork



# ----------------- 导入基本模块 -----------------
import numpy as np
import math

# ------------ 导入source中定义的函数 ------------
from source.sigmoidVector import sigmoidVector
from source.sigmoidMatrix import sigmoidMatrix
from source.sigmoidThreshold import sigmoidThreshold
    
# ----------------- 定义 base 类 -----------------
class Regression(object):
    def __init__(self, X, y, threshold = 0.5):
        self.thetas = None
        self.X = X
        self.y = y

# ---------------- 定义逻辑回归类 ----------------
class LogisticRegression(Regression):
    def __init__(self, X, y, threshold = 0.5):
        Regression.__init__(self, X, y, threshold = 0.5) # 继承 Regression 类
        self.m = 0
        self.threshold = threshold
        self.epoch = 1
    
    def fit(self, alpha = 0.01, accuracy = 0.001):
        self.thetas = np.full(self.X.shape[1] + 1,0.5)
        self.m = self.X.shape[0]
        a = np.full((self.m, 1), 1)
        Xb = np.column_stack((a,self.X))
        n  = self.X.shape[1]+1

        while True:
            before = self.costFunc(Xb, self.y)
            c = sigmoidMatrix(Xb, self.thetas) - self.y
            for j in range(n):
                self.thetas[j] = self.thetas[j] - alpha * np.sum(c * Xb[:,j])
            after = self.costFunc(Xb, self.y)
            if after == before or math.fabs(after - before) < accuracy:
                break
            self.epoch += 1
            
    def auto_alpha(self):
        if self.epoch < 100:
            return 1/math.exp(self.epoch)/100
        else:
            return 1/math.exp(100)

    def auto_fit(self, accuracy = 0.001):
        self.thetas = np.full(self.X.shape[1] + 1,0.5)
        self.m = self.X.shape[0]
        a = np.full((self.m, 1), 1)
        Xb = np.column_stack((a,self.X))
        n  = self.X.shape[1]+1
        
        while True:
            before = self.costFunc(Xb, self.y)
            c = sigmoidMatrix(Xb, self.thetas) - self.y
            for j in range(n):
                self.thetas[j] = self.thetas[j] - self.auto_alpha() * np.sum(c * Xb[:,j])
            after = self.costFunc(Xb, self.y)
            if after == before or math.fabs(after - before) < accuracy:
                break
            self.epoch += 1
            
    def costFunc(self, Xb, y):
        sum = 0.0
        for i in range(self.m):
            yPre = sigmoidVector(Xb[i,], self.thetas)
            if yPre == 1 or yPre == 0:
                return float(-2**31)
            sum += y[i] * math.log(yPre) + (1 - y[i]) * math.log(1 - yPre)
        return -1/self.m * sum
        
    def predict(self):
        a = np.full((len(self.X), 1), 1)
        Xb = np.column_stack((a, self.X))
        return sigmoidThreshold(Xb, self.thetas, self.threshold)
