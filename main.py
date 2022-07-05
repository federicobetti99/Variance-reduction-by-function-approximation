import matplotlib.pyplot as plt
from helpers import *


def f(x):
    return 1 / (25 * x ** 2 + 1)


def w(x, n):
    degs = np.arange(n+1)
    degs = degs[..., np.newaxis]
    x = x[..., np.newaxis].T
    poly = ((np.sqrt(2 * degs + 1) * eval_sh_legendre(degs, x)) ** 2).T
    return (n+1) / np.sum(poly, axis=1)


# I_exact = 1/5 * np.arctan(5)
# M_grid = 100 * np.arange(1, 11)
# errors_CV_20_s = []
# errors_CV_40_s = []
# errors_CV_40 = []
# errors_CV_20 = []
# errors_CV_sqrt_M = []
# errors_CV_sqrt_M_s = []
# conds_CV_20 = []
# conds_CV_20_s = []
# conds_CV_sqrt_M = []
# conds_CV_sqrt_M_s = []
#
# for M in M_grid:
#     _, err, cond = interpolation_estimator(10, f, M, I_exact)
#     _, errs, conds = interpolation_estimator(2, f, M, I_exact, QMC=True)
#     errors_CV_20.append(err)
#     errors_CV_20_s.append(errs)
#     conds_CV_20.append(cond)
#     conds_CV_20_s.append(conds)
    # _, err, cond = interpolation_estimator(10, f, M, I_exact)
    # _, errs, conds = interpolation_estimator(10, f, M, I_exact, stratification=True)
    # errors_CV_40.append(err)
    # errors_CV_40_s.append(errs)
    # _, err, cond = interpolation_estimator(int(np.sqrt(M)), f, M, I_exact)
    # _, errs, conds = interpolation_estimator(int(np.sqrt(M)), f, M, I_exact, QMC=True)
    # errors_CV_sqrt_M.append(err)
    # errors_CV_sqrt_M_s.append(errs)
    # conds_CV_sqrt_M.append(cond)
    # conds_CV_sqrt_M_s.append(conds)
# plt.loglog(M_grid, np.sqrt(1/M_grid), '--', label="$O(1/\sqrt{M})$")
# plt.loglog(M_grid, errors_CV_20, '--', label="Standard MCLS, n = 10")
# plt.loglog(M_grid, errors_CV_20_s, '--', label="Stratified MCLS, n = 2")
# plt.loglog(M_grid, errors_CV_40, '--', label="Standard MCLS, n = 10")
# plt.loglog(M_grid, errors_CV_40_s, '--', label="Stratified MCLS, n = 10")
# plt.loglog(M_grid, errors_CV_sqrt_M, '--', label="Standard MCLS, $n = \sqrt{M}$")
# plt.loglog(M_grid, errors_CV_sqrt_M_s, '--', label=" Stratified MCLS, $n = \sqrt{M}$")
# plt.legend(loc="best")
# plt.show()

# plt.loglog(M_grid, conds_CV_20, '--', label="Standard MCLS, n = 10")
# plt.loglog(M_grid, conds_CV_20_s, '--', label="Stratified MCLS, n = 2")
# # plt.loglog(M_grid, errors_CV_40, '--', label="Standard MCLS, n = 10")
# # plt.loglog(M_grid, errors_CV_40_s, '--', label="Stratified MCLS, n = 10")
# plt.loglog(M_grid, conds_CV_sqrt_M, '--', label="Standard MCLS, $n = \sqrt{M}$")
# plt.loglog(M_grid, conds_CV_sqrt_M_s, '--', label=" Stratified MCLS, $n = \sqrt{M}$")
# plt.legend(loc="best")
# plt.show()
