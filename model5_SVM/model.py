import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import datasets



class MySVM:
    def __init__(self, C=0.6, toler=0.001, kTup=('linear',1),maxepoches=10000):
        self.X = None
        self.labelMat = None
        self.C = C
        self.tol = toler
        self.m = 0
        self.alphas = None
        self.b = 0
        self.w=0
        self.eCache = None
        self.K =None
        self.kTup=kTup
        self.maxepoches=maxepoches

    def fit(self,dataMatIn, classLabels):
        dataMatIn=np.mat(dataMatIn)
        classLabels=np.mat(classLabels).transpose()
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.labelMat = classLabels
        self.X = dataMatIn
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelfun(self.X, self.X[i, :], self.kTup)
        epoch = 0
        entireSet = True
        alphaPairsChanged = 0
        while (epoch < self.maxepoches) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:  # go over all
                for i in range(self.m):
                    alphaPairsChanged += self.innerL(i)
                    print( "fullSet, iter: %d i:%d, pairs changed %d" % (epoch,i,alphaPairsChanged))
                epoch += 1
            else:
                nonBoundIs = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i)
                    print( "non-bound, iter: %d i:%d, pairs changed %d" % (epoch,i,alphaPairsChanged))
                epoch += 1
            if entireSet:
                entireSet = False  # toggle entire set loop
            elif (alphaPairsChanged == 0):
                entireSet = True
            print( "iteration number: %d" % epoch)
        self.w = sum(np.array(self.alphas) * np.array(self.labelMat.reshape((-1, 1))) * np.array(np.array(self.X)))

    def innerL( self,i):
        fXi = float(np.multiply(self.alphas, self.labelMat).T * self.K[:, i] + self.b)
        Ei = fXi - float(self.labelMat[i])
        if ((self.labelMat[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or (
                (self.labelMat[i] * Ei > self.tol) and (self.alphas[i] > 0)):

            maxK = -1
            maxDeltaE = 0
            Ej = 0
            j = 0
            self.eCache[i] = [1, Ei]
            validEcacheList = np.nonzero(self.eCache[:, 0].A)[0]
            if (len(validEcacheList)) > 1:
                for k in validEcacheList:
                    if k == i: continue
                    fXk = float(np.multiply(self.alphas, self.labelMat).T * self.K[:, k] + self.b)
                    Ek = fXk - float(self.labelMat[k])
                    deltaE = abs(Ei - Ek)
                    if (deltaE > maxDeltaE):
                        maxK = k;
                        maxDeltaE = deltaE
                        Ej = Ek
                j = maxK
            else:
                j = i
                while (j == i):
                    j = int(random.uniform(0, self.m))
                fXj = float(np.multiply(self.alphas, self.labelMat).T * self.K[:, j] + self.b)
                Ej = fXj - float(self.labelMat[j])

            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            if (self.labelMat[i] != self.labelMat[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                # print ("L==H")
                return 0
            eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                # print( "eta>=0")
                return 0
            self.alphas[j] -= self.labelMat[j] * (Ei - Ej) / eta
            if self.alphas[j] > H:
                self.alphas[j] = H
            elif self.alphas[j] < L:
                self.alphas[j] = L
            self.eCache[j] = [1, Ej]
            if (abs(self.alphas[j] - alphaJold) < 0.00001):
                # print( "j not moving enough")
                return 0
            self.alphas[i] += self.labelMat[j] * self.labelMat[i] * (
                        alphaJold - self.alphas[j])

            self.eCache[i] = [1, Ei]
            b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.K[i, i] - self.labelMat[j] * (
                    self.alphas[j] - alphaJold) * self.K[i, j]
            b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.K[i, j] - self.labelMat[j] * (
                    self.alphas[j] - alphaJold) * self.K[j, j]
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def predict(self,dataArrTest,labelArrTest=None):

        svInd = np.nonzero(self.alphas.A > 0)[0]
        sVs = self.X[svInd]
        labelSV = self.labelMat[svInd]
        m, n = np.shape(self.X)
        errorCount = 0
        for i in range(m):
            kernelEval = kernelfun(sVs, self.X[i, :],self.kTup)
            predict = kernelEval.T * np.multiply(labelSV, self.alphas[svInd]) + self.b
            if np.sign(predict) != np.sign(self.labelMat[i]): errorCount += 1
        print("the training error rate is: %f" % (float(errorCount) / m))
        errorCount = 0
        datMat = np.mat(dataArrTest)
        # labelMat = np.mat(labelArrTest).transpose()
        m, n = np.shape(datMat)
        pre=np.zeros(m)
        for i in range(m):
            kernelEval = kernelfun(sVs, datMat[i, :], self.kTup)
            predict = kernelEval.T * np.multiply(labelSV, self.alphas[svInd]) + self.b
            pre[i]=np.sign(predict)
        return pre

def kernelfun(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'linear':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('We did not implement this kernel function ')
    return K

def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    print(X_new.shape)
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

if __name__ == '__main__':

    # 调用主函数
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2,
                        random_state=0, cluster_std=0.60)
    #X, y = datasets.make_moons(noise=0.15, random_state=666)

    for i in range(len(y)):
        y[i] = y[i] * 2 - 1
    trainer = MySVM(0.6, 0.0001, ('linear',1), 100)

    trainer.fit(X, y)
    b2, alphas2 = trainer.b,trainer.alphas
    predict=trainer.predict(X, y)
    print(predict[2])

    w3=trainer.w
    dataArrWithAlpha2 = np.array(np.concatenate((X, alphas2), axis=1))
    print('b:', b2, '\nw:', w3, '\ndata，alphas，支撑向量:\n', dataArrWithAlpha2[
        dataArrWithAlpha2[:, -1] > 0])
    #plot_decision_boundary(trainer, axis=[-1.5, 2.5, -1.0, 1.5])
    plot_decision_boundary(trainer, axis=[-1.0, 4.0, -1.0, 6.0])
    plt.scatter(X[y==-1, 0], X[y==-1, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()
