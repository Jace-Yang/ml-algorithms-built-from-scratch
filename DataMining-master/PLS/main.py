#!/usr/bin/env python
# coding: utf-8

# 
# # 简介
# 
# 此篇意在比较 PCR（主成分分析）和 PLS（偏最小二乘回归）的效果，得出在什么情况下哪种方法更为合适。
# 
# 模拟了一个 n 个观测值，p 个变量，变量之间相关系数为 $\rho$ 的数据集，通过 $\beta_0$ 和 $\beta_1$ 加上一个标准正态分布的残差模拟出被解释变量。
# 
# # 说明
# 
# 在 comparison.py 中定义了一个 comparison 函数，用于输出 PCR 和 PLS 的指标对比，分别包括：
# 
# - 交叉验证中的测试误差
# - 成分的个数（交叉验证中的测试误差取到最小时）
# - 对Y的解释程度（在此成分个数下）
# 
# # 结论
# 
# 相比于 PCR，PLS 在以下情况的表现更佳：
# 
# - 变量个数更多
# - 变量之间相关系数较小
# - 各个变量的系数较大（变量对结果的影响较大）
# 
# # 模拟过程
# 
# ## 变化 - p
# 
# 

# In[12]:


import numpy as np
import pandas as pd
from scipy.stats import norm
from src.scale import scale
from src.sim import sim
from model.comparison import comparison


# In[13]:


n, p, rho = 1000, 10, 0.5
mu = norm.rvs(size=p, scale=1)
beta0, beta1 = 0.5, 0.5 * np.ones(p, dtype=float)
comparison(n, p, rho, mu, beta0, beta1)


# In[14]:


n, p, rho = 1000, 30, 0.5
mu = norm.rvs(size=p, scale=1)
beta0, beta1 = 0.5, 0.5 * np.ones(p, dtype=float)
comparison(n, p, rho, mu, beta0, beta1)


# In[15]:


n, p, rho = 1000, 50, 0.5
mu = norm.rvs(size=p, scale=1)
beta0, beta1 = 0.5, 0.5 * np.ones(p, dtype=float)
comparison(n, p, rho, mu, beta0, beta1)


# 
# ## 变化 - rho
# 
# 

# In[16]:


n, p, rho = 1000, 30, 0.25
mu = norm.rvs(size=p, scale=1)
beta0, beta1 = 0.5, 0.5 * np.ones(p, dtype=float)
comparison(n, p, rho, mu, beta0, beta1)


# In[17]:


n, p, rho = 1000, 30, 0.5
mu = norm.rvs(size=p, scale=1)
beta0, beta1 = 0.5, 0.5 * np.ones(p, dtype=float)
comparison(n, p, rho, mu, beta0, beta1)


# In[18]:


n, p, rho = 1000, 30, 0.75
mu = norm.rvs(size=p, scale=1)
beta0, beta1 = 0.5, 0.5 * np.ones(p, dtype=float)
comparison(n, p, rho, mu, beta0, beta1)


# 
# ## 变化 - beta
# 
# 

# In[19]:


n, p, rho = 1000, 30, 0.5
mu = norm.rvs(size=p, scale=1)
beta0, beta1 = 0.1, 0.1 * np.ones(p, dtype=float)
comparison(n, p, rho, mu, beta0, beta1)


# In[20]:


n, p, rho = 1000, 30, 0.5
mu = norm.rvs(size=p, scale=1)
beta0, beta1 = 0.5, 0.5 * np.ones(p, dtype=float)
comparison(n, p, rho, mu, beta0, beta1)


# In[21]:


n, p, rho = 1000, 30, 0.5
mu = norm.rvs(size=p, scale=1)
beta0, beta1 = 1, 1 * np.ones(p, dtype=float)
comparison(n, p, rho, mu, beta0, beta1)

