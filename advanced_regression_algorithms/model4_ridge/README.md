# Introduction

One way the predictive capability of an OLS model can be hampered is by overfitting the training set data. When training on such a sample, a model can become overfit when it uses patterns and relationships found within the sample. 

Ridge gives a predictive model with **high variance** (parameter predictions vary greatly from sample to sample) and **low bias** (error introduced by the modelling technique), which is sub-optimal in its capabilities. It can be achieved through biasing parameter predictions towards zero (shrinkage). Ridge Regression (also known as Tikhonov regularization or $L_2$-regularization) achieves this shrinkage of the OLS predictions through the addition of an $L_2$ penalty to the loss function. 

![png](./image/output_12_0.png)

