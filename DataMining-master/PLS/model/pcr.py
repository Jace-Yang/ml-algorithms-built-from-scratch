import pandas as pd
import numpy as np
from src.pcr_pure import pcr_pure


class PCR(object):
    def __init__(self, x, y, x_names, is_scale=True, is_var_exp=True):
        self.x = x
        self.y = y
        self.n, self.p = np.shape(x)
        self.xtx = np.dot(x.T, x)
        self.xty = np.dot(x.T, y)
        self.yty = np.sum(y * y)
        self.coe = 0
        self.b = np.array([0])
        self.is_scale = is_scale
        self.x_mean = np.mean(x, axis=0)
        self.y_mean = np.mean(y)
        self.cv_err = 0
        self.cv_b = 0
        self.is_var_exp = is_var_exp
        self.var_exp_y = None
        self.var_exp_x = None
        self.x_names = x_names

    def pcr(self):
        if self.is_var_exp:
            yty = self.yty
        else:
            yty = None
        self.b, self.var_exp_x, self.var_exp_y = pcr_pure(self.xtx, self.xty, self.x_mean, self.y_mean, self.n,
                                                          self.p, self.is_scale, self.is_var_exp, yty)

    def cv(self, k=10):
        indexs = np.array_split(np.random.permutation(np.arange(0, self.n)), k)

        def cvk(index):
            tx = self.x[index]
            tn, tp = np.shape(tx)
            if tn == 1:
                tx = tx.reshape((1, self.p))
            tn_ = self.n - tn
            ty = self.y[index]
            txt = tx.T
            txx_ = self.xtx - np.dot(txt, tx)
            txy_ = self.xty - np.dot(txt, ty)
            tx_sum = np.sum(tx, axis=0)
            ty_sum = np.sum(ty)
            tx_mean_ = (self.n * self.x_mean - tx_sum) / tn_
            ty_mean_ = (self.n * self.y_mean - ty_sum) / tn_
            tb = pcr_pure(txx_, txy_, tx_mean_, ty_mean_, tn_, tp, self.is_scale)
            tx = np.c_[np.ones((tn, 1)), tx]
            ty_pred = np.dot(tb, tx.T)
            err = ty_pred - ty
            err = err * err
            return np.sum(err, axis=1)
        self.cv_err = np.sum(np.array([cvk(index) for index in indexs]), axis=0) / self.n
        min_k = np.argmin(self.cv_err)
        self.cv_b = self.b[min_k]
        return self.cv_b

    def report_coe(self):
        names = np.append("inter", self.x_names)
        results = pd.DataFrame(self.b, columns=names, index=np.arange(1, self.p+1))
        results["cverr"] = self.cv_err
        return results

    def report_var_exp(self):
        var_exp = np.c_[self.var_exp_x, self.var_exp_y]
        results = pd.DataFrame(var_exp, columns=["var_exp_x", "var_exp_y"], index=np.arange(self.p))
        return results

    def predict(self, xn):
        tn, _ = np.shape(xn)
        xn_ = np.c_[np.ones((tn, 1)), xn]
        return np.dot(self.cv_b, xn_.T)

    def predict_err(self, xn, yn):
        err = yn - self.predict(xn)
        err = err * err
        return np.mean(err)

    def test_err(self, xn, yn):
        tn, _ = np.shape(xn)
        xn_ = np.c_[np.ones((tn, 1)), xn]
        err = yn - np.dot(self.b, xn_.T, )
        err = err * err
        err_mean = np.mean(err, axis=1)
        err_std = np.std(err, axis=1, ddof=1)/np.sqrt(tn)
        result = {'err_mean': err_mean, 'err_std': err_std}
        return pd.DataFrame(result)
