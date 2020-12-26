import numpy as np
from source.plsr_pure import pls_pure
import pandas as pd


class PLSR(object):
    def __init__(self, x, y, x_names, is_scale=True, is_var_exp=True):
        self.x = x
        self.y = y
        self.names = x_names
        self.is_scale = is_scale
        self.is_var_exp = is_var_exp
        self.n, self.p = np.shape(x)
        self.xty = np.dot(x.T, y)
        self.xtx = np.dot(x.T, x)
        self.yty = np.dot(y.T, y)
        self.x_mean = np.mean(x, axis=0)
        self.y_mean = np.mean(y)
        self.b = np.array([0])
        self.b_min_cv = 0
        self.var_x_exp = None
        self.var_y_exp = None
        self.cv_error = None

    def plsr(self):
        if self.is_var_exp:
            yty = self.yty
        else:
            yty = None
        self.b, self.var_x_exp, self.var_y_exp = pls_pure(self.xtx, self.xty, self.x_mean, self.y_mean, self.n, self.p,
                                                          self.is_scale, self.is_var_exp, yty)

    def cv_kfold(self, k=10):
        indices = np.array_split(np.random.permutation(np.arange(0, self.n)), k)

        def cv(index):
            x_test = self.x[index]
            y_test = self.y[index]
            n_test, p_test = np.shape(x_test)
            n_train = self.n - n_test
            xt_test = x_test.T
            xtx_train = self.xtx - np.dot(xt_test, x_test)
            xty_train = self.xty - np.dot(xt_test, y_test)
            x_train_mean = (self.x_mean * self.n - np.sum(x_test, axis=0)) / n_train
            y_train_mean = (self.y_mean * self.n - np.sum(y_test)) / n_train
            b_cv = pls_pure(xtx_train, xty_train, x_train_mean, y_train_mean, self.is_scale, is_var_exp=False)
            x_test = np.c_[np.ones((n_test, 1)), x_test]
            y_pred = np.dot(b_cv, x_test.T)
            cv_error = y_pred - y_test
            cv_error = cv_error * cv_error
            return np.sum(cv_error, axis=1)

        self.cv_error = np.sum(np.array([cv(index) for index in indices]), axis=0) / self.n
        index_min_error = np.argmin(self.cv_error)
        self.b_min_cv = self.b[index_min_error]
        return self.b_min_cv

    def coefficient(self):
        names = np.append('intercept', self.names)
        output = pd.DataFrame(self.b, index=np.arange(1, self.p+1), columns=names)
        output['cv_error'] = self.cv_error
        return output

    def var_explained(self):
        var_exp = np.c_[self.var_x_exp, self.var_y_exp]
        output = pd.DataFrame(var_exp, index=np.arange(1, self.p+1), columns=['var_x_explained', 'var_y_explained'])
        return output

    def predict(self, x_test):
        xn, _ = np.shape(x_test)
        x_test = np.c_[np.ones((xn, 1)), x_test]
        return np.dot(self.b_min_cv, x_test.T)

    def predict_error(self, x_test, y_test):
        y_pred = self.predict(x_test)
        error = y_pred - y_test
        error = np.mean(error * error)
        return error

    def all_error(self, x_test, y_test):
        xn, _ = np.shape(x_test)
        x_test = np.c_[np.ones((xn, 1)), x_test]
        y_pred = np.dot(self.b, x_test.T)
        error = y_pred - y_test
        error_mean = np.mean(error * error, axis=1)
        error_std = np.std(error, axis=1, ddof=1) / np.sqrt(xn)
        output = pd.DataFrame({'err_mean': error_mean, 'err_std': error_std})
        return output