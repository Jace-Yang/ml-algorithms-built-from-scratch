import numpy as np
import pandas as pd
from source.pcr_pure import pcr_pure


class PCR(object):
    def __init__(self, x, y, x_names, is_scale=True, is_var_exp=True):
        self.x = x
        self.y = y
        self.n, self.p = np.shape(x)
        self.x_names = x_names
        self.xtx = np.dot(x.T, x)
        self.xty = np.dot(x.T, y)
        self.yty = np.sum(y * y)
        self.is_scale = is_scale
        self.is_var_exp = is_var_exp
        self.b = np.array([0])
        self.var_x_exp = None
        self.var_y_exp = None
        self.x_mean = np.mean(x, axis=0)
        self.y_mean = np.mean(y)
        self.min_b_cv = 0
        self.cv_err = 0

    def pcr(self):
        if self.is_var_exp:
            yty = self.yty
        else:
            yty = None
        self.b, self.var_x_exp, self.var_y_exp = pcr_pure(self.xtx, self.xty, self.x_mean, self.y_mean, self.n, self.p,
                                                          self.is_scale, self.is_var_exp, yty)

    def coeff(self):
        names = np.append('intercept', self.x_names)
        output = pd.DataFrame(self.b, index=np.arange(1, self.p+1), columns=names)
        return output

    def cv_kfold(self, k_fold=10):
        indices = np.array_split(np.random.permutation(np.arange(0,self.n)), k_fold)

        def cv(index):
            x_cv = self.x[index]
            y_cv = self.y[index]
            xtx_cv = self.xtx - np.dot(x_cv.T, x_cv)
            xty_cv = self.xty - np.dot(x_cv.T, y_cv)
            n_cv, p_cv = np.shape(x_cv)
            if n_cv == 1:
                x_cv = x_cv.reshape((1,self.p))
            n_cv_t = self.n - n_cv
            x_mean_cv = (self.n*self.x_mean - np.sum(x_cv, axis=0))/n_cv_t
            y_mean_cv = (self.n*self.y_mean - np.sum(y_cv))/n_cv_t
            b_cv = pcr_pure(xtx_cv, xty_cv, x_mean_cv, y_mean_cv, n_cv_t, p_cv, self.is_scale, is_var_exp=False)

            x_cv = np.c_[np.ones((n_cv, 1)), x_cv]
            y_pre_cv = np.dot(b_cv, x_cv.T)
            cv_err = np.sum((y_pre_cv-y_cv)*(y_pre_cv-y_cv), axis=1)
            return cv_err

        self.cv_err = np.sum(np.array([cv(index) for index in indices]), axis=0)/self.n
        min_err_index = np.argmin(self.cv_err)
        self.min_b_cv = self.b[min_err_index]
        return self.min_b_cv
    
    def predict(self, x_test):
        n_test, _ = np.shape(x_test)
        x_test = np.c_[np.ones((n_test,1)), x_test]
        y_pred = np.dot(self.min_b_cv, x_test.T)
        return y_pred

    def predict_err(self, x_test, y_test):
        error = y_test - self.predict(x_test)
        error = np.mean(error * error)
        return error

    def var_exp(self):
        var_exp = np.c_[self.var_x_exp, self.var_y_exp]
        output = pd.DataFrame(var_exp, index=np.arange(self.p), columns=['var_x_explained', 'var_y_explained'])
        return output

    def all_error(self, x_test, y_test):
        n_test, _ = np.shape(x_test)
        x_test = np.c_[np.ones((n_test, 1)), x_test]
        err = y_test - np.dot(self.b, x_test.T)
        err = err * err
        err_mean = np.mean(err, axis=1)
        err_std = np.std(err, axis=1, ddof=1)/np.sqrt(n_test)
        output = pd.DataFrame({'err_mean': err_mean, 'err_std': err_std})
        return output

