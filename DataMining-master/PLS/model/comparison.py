import numpy as np
import pandas as pd
from scipy.stats import norm
from src.scale import scale
from src.sim import sim
from model.pcr import PCR
from model.pls import PLS

def comparison(n, p, rho, mu, beta0, beta1):
    is_scale = False
    is_var_exp = True
    x, y = sim(n, p, rho, mu, beta0, beta1)
    n1 = 2000
    x1, y1 = sim(n, p, rho, mu, beta0, beta1)
    x, y = sim(n, p, rho, mu, beta0, beta1)
    names = list(range(p))
    
    pcr1 = PCR(x, y, names, is_scale, is_var_exp)
    pcr1.pcr()
    pcr1.cv(n)
    pcr1.report_var_exp()
    pcr1.predict_err(x1, y1)
    frame1 = pcr1.test_err(x1, y1)["err_mean"]
    num1 = frame1.argmin()
    exp1 = float(pcr1.report_var_exp()["var_exp_y"][num1: num1 + 1].values)
    
    pls2 = PLS(x, y, names, is_scale, is_var_exp)
    pls2.pls()
    pls2.cv(n)
    pls2.report_var_exp()
    pls2.predict_err(x1, y1)
    frame2 = pls2.test_err(x1, y1)["err_mean"]
    num2 = frame2.argmin()
    exp2 = float(pls2.report_var_exp()["var_exp_y"][num2: num2 + 1].values)
    
    methods = ["PCR", "PLS"]
    components = [num1, num2]
    test_err = [min(frame1), min(frame2)]
    var_exp = [exp1, exp2]
    
    result = {"methods": methods, "n components": components, "test error": test_err, "variation explanation": var_exp}
    return pd.DataFrame(result)
