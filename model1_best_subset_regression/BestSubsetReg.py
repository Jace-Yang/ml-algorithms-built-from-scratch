import os
import sys
from scipy import linalg
from scipy.stats import norm
import numpy as np
import pandas as pd

class BestSubsetReg(object):
    def __init__(self, X, Y, names, inter = True, isCp = True, isAIC = True, isCV = True):
        self.n, self.p = X.shape
        self.names = names
        
        ## 1、截距项和模型评价标准选择
        if inter:
            self.inter = True
            self.X = np.c_[(np.ones((self.n,1))),X]
            self.p += 1
        else:
            self.inter = False
            self.X = X
        
        self.isCp = isCp
        self.isAIC = isAIC
        self.isCV = isCV

        ## 2、全模型回归需要的变量选择
        def turnbits(p, is_inter):
            def turnbits_rec(p):
                if p==1:
                    return np.array([[True, False],[True, True]])
                else:
                    tmp1 = np.c_[ turnbits_rec(p-1), np.array([False]*(2**(p-1))).reshape((2**(p-1),1))]
                    tmp2 = np.c_[ turnbits_rec(p-1), np.array([True]*(2**(p-1))).reshape((2**(p-1),1))]
                    return np.r_[tmp1, tmp2]
            if is_inter:
                return turnbits_rec(self.p-1)
            else:
                return turnbits_rec(self.p)[1:,1:] #不要第一列是因为这个是有intercept的情况，不要第一行是因为第一行全是False会出错
        self.ind_l = np.array(turnbits(self.p,self.inter))
        self.b_l = []
        self.RSS_l = []
        self.d_l = np.sum(self.ind_l, axis = 1) #所有模型参数个数d组成的列表
    
        ## 3、计算回归所需矩阵
        self.Y = Y
        self.sigma_hat_2 = 0
        self.XTX = np.dot(self.X.T, self.X)
        self.XTY = np.dot(self.X.T, self.Y)
        
    # 第一步、执行所有回归并储存结果
    def solve_sym(self,xtx, xty):
        L = np.linalg.cholesky(xtx)
        Lb = linalg.solve_triangular(L,xty,lower=True)
        return linalg.solve_triangular(L.T,Lb)
    
    def all_reg(self):
        self.b_l = [self.solve_sym(self.XTX[ind][:,ind], self.XTY[ind]) for ind in self.ind_l]      
        YTY = np.sum(self.Y**2)
        Y_hatTY_hat = [np.sum(np.power(np.dot(self.XTX[ind][:,ind],b),2)) for ind, b in zip(self.ind_l,self.b_l)]
        self.RSS_l = [np.sum((self.Y - np.dot(self.X[:,ind],b))**2) for ind, b in zip(self.ind_l,self.b_l)]
        self.sigma_hat_2 = min(self.RSS_l)/(self.n-self.p)
        
    # 第二步、计算全模型列表里各个模型的评估指标
    def run_Cp(self):
        self.Cp_l = (self.RSS_l + 2*self.d_l*self.sigma_hat_2 ) / self.n
        self.print_result("Cp",self.Cp_l)
    
    def run_AIC(self):
        self.AIC_l = self.n * np.log(self.RSS_l) + 2*self.d_l
        self.print_result("AIC",self.AIC_l)
            
    def run_CV(self, K = 10):
        np.random.seed(514)
        test_l = np.array_split(np.random.permutation(range(0,self.n)),K)
        def CV(ind,test):
            ## 测试集
            X_1 = self.X[test][:,ind]
            Y_1 = self.Y[test]
            ## 训练集
            X_0 = np.delete(self.X[:,ind], test, axis=0)
            Y_0 = np.delete(self.Y, test)
            XTX_0 = np.dot(X_0.T, X_0)
            XTY_0 = np.dot(X_0.T, Y_0)
            ## 模型求解
            b = self.solve_sym(XTX_0, XTY_0)
            ## 模型预测训练集
            Y_frcst = np.dot(X_1,b)
            ## CV结果返回
            return np.sum((Y_1 - Y_frcst)**2)
        self.CV_l = [sum([CV(ind,test) for test in test_l])/self.n for ind in self.ind_l]
            #sum 不能改成mean，因为mean会除的是K
        self.print_result("CV",self.CV_l)

#             for ind in self.ind_l:
#                 CV_l_ind = []
#                 for test in test_l:
#                     CV_l_ind.append(CV(ind, test))
#                 self.CV_l.append(sum(CV_l_ind)/self.n)
    
    def print_result(self,eval_type,value):
        prmt = np.array(self.names)
        print("—————Based on",eval_type,"———————")
        min_id = np.argmin(value)
        if self.inter:
            prmt_temp = prmt[self.ind_l[min_id]]
            prmt = np.append(np.array(['intercept']),prmt_temp)
        else: 
            print(prmt)
            print(self.ind_l[min_id])
            prmt = prmt[self.ind_l[min_id]]

        b_best = self.b_l[min_id]
        df = pd.DataFrame(b_best,prmt,columns = ["β"])
        print(df)
        print("—————————————————————————")
        print(eval_type," =",value[min_id])
        print("—————————————————————————")
        print("")
            
    # 第三步、打印每个准则下的结果
    def results(self):
        self.all_reg()
        if self.isCp:
            self.run_Cp()
        if self.isAIC:
            self.run_AIC()
        if self.isCV:
            self.run_CV()