# Best Subset Regression from scratch

## Output from the [notebook](BestSubsetRegClass.ipynb):

### 1. Packced code


```python
import os
import sys
from scipy import linalg
from scipy.stats import norm
import numpy as np
import pandas as pd

class BestSubsetReg(object):
    
    def __init__(self, X, Y, inter = True, isCp = True, isAIC = True, isCV = True):
        
        self.n, self.p = X.shape
        
        ## 1. switch on if there is an intercept term in model
        if inter:
            self.inter = True
            self.X = np.c_[(np.ones((self.n,1))),X]
            self.p += 1
        else:
            self.inter = False
            self.X = X
        
        self.isCp = self.isAIC = self.isCV = False
        
        if isCp:
            self.isCp = True

        if isAIC:
            self.isAIC = True

        if isCV:
            self.isCV = True

        ## 2、Variable selection for full model regression
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
                return turnbits_rec(self.p)[1:,1:]
        self.ind_l = turnbits(self.p,self.inter)
        self.b_l = []
        self.RSS_l = []
        self.d_l = np.sum(self.ind_l, axis = 1) #List out all possible model parameters 
    
        ## 3、Calculate the matrix needed for regression
        self.Y = Y
        self.sigma_hat_2 = 0
        self.XTX = np.dot(self.X.T, self.X)
        self.XTY = np.dot(self.X.T, self.Y)
        
    # Step 1.  Run all regressions and save the results
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
        
    # Step 2. calculate the evaluation index of each model in the whole model list
    def run_Cp(self):
        if self.isCp:
            self.Cp_l = (self.RSS_l + 2*self.d_l*self.sigma_hat_2 ) / self.n
            self.print_result("Cp",self.Cp_l)
    
    def run_AIC(self):
        if self.isAIC:
            self.AIC_l = self.n * np.log(self.RSS_l) + 2*self.d_l
            self.print_result("AIC",self.AIC_l)
            
    def run_CV(self, K = 10, seed = 514):
        if self.isCV:
            np.random.seed(seed)
            test_l = np.array_split(np.random.permutation(range(0,self.n)),K)
            def CV(ind,test):
                ## test set
                X_1 = self.X[test][:,ind]
                Y_1 = self.Y[test]
                ## train set
                X_0 = np.delete(self.X[:,ind], test, axis=0)
                Y_0 = np.delete(self.Y, test)
                XTX_0 = np.dot(X_0.T, X_0)
                XTY_0 = np.dot(X_0.T, Y_0)
                ## Solve the model
                b = self.solve_sym(XTX_0, XTY_0)
                ## Fit the training set
                Y_frcst = np.dot(X_1,b)
                ## Return cross-validation result
                return np.sum((Y_1 - Y_frcst)**2)
            self.CV_l = [sum([CV(ind,test) for test in test_l])/self.n for ind in self.ind_l]
            self.print_result("CV",self.CV_l)

#             for ind in self.ind_l:
#                 CV_l_ind = []
#                 for test in test_l:
#                     CV_l_ind.append(CV(ind, test))
#                 self.CV_l.append(sum(CV_l_ind))
    
    # helper function of just printing 1 result
    def print_result(self,eval_type,value):
            prmt = names
            print("—————Based on",eval_type,"———————")
            min_id = np.argmin(value)
            if self.inter:
                prmt_temp = prmt[self.ind_l[min_id][1:]]
                prmt = np.append(np.array(['intercept']),prmt_temp)
            else: 
                prmt = names[self.ind_l[min_id]]
            b_best = self.b_l[min_id]
            df = pd.DataFrame(b_best,prmt,columns = ["β"])
            print(df)
            print("—————————————————————————")
            print(eval_type," =",value[min_id])
            print("—————————————————————————")
            print("")
            
    # Step 3. Print result under 3 different criteria
    def print_results(self,names):
        self.all_reg()
        self.run_Cp()
        self.run_AIC()
        self.run_CV()
```

### 2 Test

#### 2.1 Load a testing data


```python
x = np.loadtxt("./data/x.txt", delimiter=",")
y = np.loadtxt("./data/y.txt", delimiter=",")
names = np.loadtxt("./data/names.txt", delimiter=",", dtype=str)[0:8] # the name of x variables
```

#### 2.2 Result

##### With Intercept:


```python
reg1 = BestSubsetReg(x, y)
reg1.print_results(names)
```

    —————Based on Cp ———————
                      β
    intercept  0.494729
    lcavol     0.543998
    lweight    0.588213
    age       -0.016445
    lbph       0.101223
    svi        0.714904
    —————————————————————————
    Cp  = 0.5186421068394103
    —————————————————————————
    
    —————Based on AIC ———————
                      β
    intercept  0.494729
    lcavol     0.543998
    lweight    0.588213
    age       -0.016445
    lbph       0.101223
    svi        0.714904
    —————————————————————————
    AIC  = 380.0243336247493
    —————————————————————————
    
    —————Based on CV ———————
                      β
    intercept -0.777157
    lcavol     0.525852
    lweight    0.661770
    svi        0.665667
    —————————————————————————
    CV  = 0.5269161154609794
    —————————————————————————
    


##### With no Intercept:


```python
reg2 = BestSubsetReg(x, y, inter=False, isAIC = False)
reg2.print_results(names)
```

    —————Based on Cp ———————
                    β
    lcavol   0.536674
    lweight  0.661742
    age     -0.012716
    lbph     0.085012
    svi      0.705686
    —————————————————————————
    Cp  = 0.5096011437549013
    —————————————————————————
    
    —————Based on CV ———————
                    β
    lcavol   0.532744
    lweight  0.440686
    lbph     0.090976
    svi      0.713390
    —————————————————————————
    CV  = 0.5124266762953688
    —————————————————————————
    

