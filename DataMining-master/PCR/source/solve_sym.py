from scipy import linalg

def solve_sym(xtx, xty):
    L = linalg.cholesky(xtx)
    return linalg.lapack.dpotrs(L, xty)[0] 
