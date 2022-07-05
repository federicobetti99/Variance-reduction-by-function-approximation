# IMPORT LIBRARIES
import numpy as np
import scipy.stats as st
# the package eval_sh_legendre works by default on the interval [0,1]
from scipy.special import eval_sh_legendre
# import QMC libraries from laboratory 9 to compute QMCLS estimates
from sobol_new import *


# FUNCTIONS TO BE USED FOR THE ESTIMATIONS OF THE 1D INTEGRAL

# CRUDE MONTE CARLO ESTIMATOR FOR THE INTEGRAL OF g
def crude_monte_carlo(g, M, I_exact):
    """
    This function computes a CMC estimate of I_exact
    :param g: the function to be integrated
    :param M: the number of samples
    :param I_exact: the exact value of the integral
    :return: the value of the CMC estimate and the true error with respect to I_exact
    """
    # draw M uniform points
    U = st.uniform.rvs(size=M)
    # compute the CMC estimator
    estimate_ = np.mean(g(U))
    # compute the error with respect to the exact value
    error_ = np.abs(I_exact - estimate_)
    return estimate_, error_


# RANDOMIZED MIDPOINT QUADRATURE ESTIMATOR FOR THE INTEGRAL OF g
def randomized_midpoint(g, M, I_exact):
    """
    This function computes a randomized midpoint quadrature estimator for I_exact
    :param g: the function to be integrated
    :param M: the number of strata and thus of samples (1 point per stratum)
    :param I_exact: the exact value of the integral
    :return: the value of the stratified estimator and the true error with respect to I_exact
    """
    # draw a uniform random variable in each stratum
    U = st.uniform.rvs(loc=np.arange(M) / M, scale=1/M)
    # compute the stratified estimator
    estimate_ = np.mean(g(U))
    # compute the error with respect to the exact value
    error_ = np.abs(I_exact - estimate_)
    return estimate_, error_


# MCLS ESTIMATOR WITHOUT IMPORTANCE SAMPLING FOR THE INTEGRAL OF g (UNIFORMLY OR QMC DRAWN SAMPLES)
def interpolation_estimator(n, g, M, I_exact, QMC=False):
    """
    This function computes the control variate MCLS estimator of I_exact
    after interpolation of f via Legendre polynomials
    :param QMC: a boolean whether the samples are drawn in a QMC fashion by stratification
    :param n: the maximum degree of the Legendre polynomials
    :param g: the function to integrated
    :param M: the number of samples
    :param I_exact: the exact value of the integral
    :return: the value of the MCLS estimator, the true error with respect to I_exact and the conditioning of
             the Vandermonde matrix involved in solving for the regression coefficients
    """
    # construct a random grid of points in [0,1]
    if QMC:
        # draw samples in a QMC fashion stratifying [0,1] (low star discrepancy set)
        x_grid = st.uniform.rvs(loc=np.arange(M) / M, scale=1/M)
    else:
        # draw uniformly samples
        x_grid = st.uniform.rvs(size=M)
    # construct the Vandermonde matrix, normalization of the polynomials with \sqrt{2*i+1}
    degs = np.arange(n+1)
    degs = degs[..., np.newaxis]
    x_grid = x_grid[..., np.newaxis]
    V = (np.sqrt(2 * degs + 1) * eval_sh_legendre(degs, x_grid.T)).T
    # compute the conditioning of the Vandermonde matrix
    cond = np.linalg.cond(V)
    # solve the normal equations to find the optimal coefficients for the regression
    if n + 1 <= M:
        # if V has full column rank calculate the reduced QR factorization of V ---> reduced to solving a
        # triangular system
        Q, R = np.linalg.qr(V)
        c = np.linalg.solve(R, Q.T @ g(x_grid))
    else:
        # if V doesn't have full column rank call the solver
        c = np.linalg.solve(V.T @ V, V.T @ g(x_grid))
    # compute the estimator
    estimate_ = c[0]
    # compute the error with respect to the exact value
    error_ = np.abs(I_exact - estimate_)
    return estimate_, error_, cond


# ACCEPTANCE REJECTION METHOD TO DRAW SAMPLES FROM 1/w
def sampling_from_legendre(n, M, w):
    """
    This function performs an acceptance-rejection method to sample from the distribution 1/w using the arcsine
    distribution as the proposal in [0,1] until M samples aren't accepted
    :param n: the maximum degree of the Legendre polynomials
    :param M: the number of samples
    :param w: sampling from the distribution 1/w
    :return: the M samples distributed according to 1/w and the acceptance rate
    """
    # initialize the vector of the samples and the counter
    X = np.array([])
    count = 0
    while X.shape[0] < M:
        # increase by M the number of trials
        count += M
        # draw M samples distributed according to the proposal distribution
        Y = st.arcsine.rvs(size=M)
        # draw M uniform samples
        U = st.uniform.rvs(size=M)
        # get the accepted samples
        Accepted = np.where(U <= 1 / (w(Y, n) * 4 * np.exp(1) * st.arcsine.pdf(Y)))[0]
        print("Accepted", len(Accepted), "out of", M)
        # a check on the number of final samples accepted during the last iteration, due to the expected acceptance rate
        # which is approximately 0.09 it is reasonable to assume that we don't waste too many samples here
        if len(Accepted) >= M-X.shape[0]:
            Accepted = Accepted[:(M-X.shape[0])]
            print("Constraint, taken only", M-X.shape[0])
        # concatenate the accepted samples to the already existing ones
        X = np.concatenate([X, Y[Accepted]])
        print("Concatenate", len(Accepted), "new samples")
        print("Currently we have", len(X), "samples")
    return X, M/count


# IMPORTANCE SAMPLING MCLS ESTIMATOR FOR THE INTEGRAL OF g (SAMPLES DRAWN FROM THE DISTRIBUTION 1/w)
def importance_sampling_estimator(n, g, w, M, I_exact):
    """
    This function computes the importance sampling MCLS estimator to solve conditioning problems for n = n(M)
    :param n: the maximum degree of the Legendre polynomials
    :param g: the function to be integrated
    :param w: 1/w is the distribution of the samples
    :param M: the number of samples
    :param I_exact: the exact value of the integral
    :return: the value of the importance sampling MCLS estimator, the true error with respect to I_exact,
    the conditioning of the weighted Vandermonde matrix involved in solving for the regression coefficients
    and the acceptance rate of the acceptance-rejection algorithm to draw samples distributed as 1/w
    """
    # draw the samples obtained by acceptance-rejection
    Y, accpt = sampling_from_legendre(n, M, w)
    # compute the weight matrix
    W = np.diag(w(Y, n))
    # compute the Vandermonde matrix, normalization of the Legendre polynomials with \sqrt{2*i+1}
    degs = np.arange(n + 1)
    degs = degs[..., np.newaxis]
    Y = Y[..., np.newaxis]
    V = (np.sqrt(2 * degs + 1) * eval_sh_legendre(degs, Y.T)).T
    # compute the conditioning of the weighted Vandermonde matrix
    cond = np.linalg.cond(np.sqrt(W) @ V)
    # solve the normal equations to find the optimal coefficients for the regression
    if n + 1 <= M:
        # if \sqrt{W} V has full column rank calculate its reduced QR factorization ---> reduced to solving a
        # triangular system
        Q, R = np.linalg.qr(np.sqrt(W) @ V)
        c = np.linalg.solve(R, Q.T @ np.sqrt(W) @ g(Y))
    else:
        # if V doesn't have full column rank call the solver
        c = np.linalg.solve(V.T @ W @ V, V.T @ W @ g(Y))
    # compute the importance sampling estimator
    estimate_ = c[0]
    # compute the error with respect to the exact value
    error_IS = np.abs(I_exact - estimate_)
    return estimate_, error_IS, cond, accpt


# FUNCTIONS TO BE USED FOR THE ESTIMATIONS OF THE 2D INTEGRAL

# SOLVING THE FITZHUGH-NAGUMO MODEL FOR SOME FIXED VALUES OF a AND b BY EXPLICIT EULER
def solve_Fitzhugh_Nagumo(a, b, dt, T, v0=0, w0=0, epsilon=0.08, I_=1):
    """
    This function solves the Fitzhugh-Nagumo system using the Explicit Euler Method
    :param a: a uniform random variable
    :param b: a uniform random variable
    :param dt: the integration time step for the ODE
    :param T: the final time of integration for the ODE
    :param v0: the initial condition on v
    :param w0: the initial condition on w
    :param epsilon: a parameter of the system
    :param I_: a parameter of the system
    :return: the numerical solution (v(t),w(t)) of the ODE
    """
    # initialization
    v = np.zeros(int(T/dt))
    w = np.zeros(int(T/dt))
    v[0] = v0
    w[0] = w0
    for i in range(len(v)-1):
        # compute the right hand side making the change of variable in a and b
        rhs = [v[i] - v[i] ** 3 / 3 - w[i] + I_, epsilon*(v[i] + 1/5*a+3/5 - (1/5*b+0.7)*w[i])]
        # Explicit Euler step
        v[i+1] = v[i] + dt * rhs[0]
        w[i+1] = w[i] + dt * rhs[1]
    return v, w


# CALCULATE THE QUANTITY OF INTEREST Q
def QoI(a, b, dt, T, v0=0, w0=0, epsilon=0.08, I_=1):
    """
    This function computes the quantity of interest
    :param a: a uniform random variable
    :param b: a uniform random variable
    :param dt: the integration time step for the ODE
    :param T: the final time of integration for the ODE
    :param v0: the initial condition on v
    :param w0: the initial condition on w
    :param epsilon: a parameter of the system
    :param I_: a parameter of the system
    :return: the quantity of interest Q
    """
    # get the numerical solution v
    v_, _ = solve_Fitzhugh_Nagumo(a, b, dt, T, v0, w0, epsilon, I_)
    # compute Q
    Q = 0.04 * dt / T * np.sum((v_[:-1] ** 2 + v_[1:] ** 2) / 2)
    return Q


# COMPUTE A CRUDE MONTE CARLO ESTIMATE OF THE AVERAGE OF Q
def crude_monte_carlo_Q(M, dt, T, Q_ref):
    """
    This function computes a CMC estimate of the average of Q
    :param M: the number of samples
    :param dt: the integration time step for the ODE
    :param T: the final time of integration for the ODE
    :param Q_ref: the reference value for the average of Q
    :return: the CMC estimate and the error with respect to Q_ref
    """
    # draw M uniform points in [0,1]^2
    X = st.uniform.rvs(size=(2, M))
    # calculate the quantity of interest for each sample
    Q = np.array([QoI(X[0, i], X[1, i], dt, T) for i in range(M)])
    # compute the estimator
    estimate_ = np.mean(Q)
    # compute the error
    error_ = np.abs(Q_ref - estimate_)
    return estimate_, error_


# CONSTRUCTION OF THE VANDERMONDE MATRIX FOR THE 2d CASE (UNIFORMLY OR QMC DRAWN SAMPLES)
def basis_2d_legendre(k, M, QMC=False):
    """
    This function computes the Vandermonde Matrix for 2d Legendre polynomials taking only products that are less of
    degree k combined
    :param QMC: a boolean whether the samples are drawn in a QMC fashion using the Sobol sequence
    :param k: the highest degree of the Legendre polynomials, so that the resulting Vandermonde matrix
              will have dimension M,n where n = (k+1)*(k+2)/2
    :param M: the number of samples
    :return: the Vandermonde matrix, the samples and the number of degrees of freedom
             (i.e. columns of the Vandermonde matrix)
    """
    if QMC:
        # draw the samples in a QMC fashion using the Sobol sequence
        X = generate_points(M, 2, 0).T
    else:
        # draw uniformly samples
        X = st.uniform.rvs(size=(2, M))
    # Number of basis functions of degree <= k in the two dimensional case
    n = np.int32((k+1)*(k+2)/2)
    # initialize the Vandermonde matrix
    V = np.zeros((M, n))
    # initialize the counter
    count = 0
    for i in range(k + 1):
        for j in range(k + 1):
            # if the sum of the degrees is below k
            if i + j <= k:
                # construct the count-th column of the V matrix by multiplying the basis polynomials of degree i and j
                V[:, count] = np.sqrt(2*i + 1) * np.sqrt(2*j + 1) \
                              * eval_sh_legendre(i, X[0, :]) * eval_sh_legendre(j, X[1, :])
                count += 1
    return V, X, n


# MCLS ESTIMATORS FOR THE AVERAGE OF Q (UNIFORMLY OR QMC DRAWN SAMPLES)
def interpolation_estimator_Q(k, M, dt, T, Q_ref, alpha=0.05, QMC=False):
    """
    This function computes the control variate estimator of the average of Q and estimates the error
    :param QMC: a boolean whether the samples are drawn in a QMC fashion using the Sobol sequence
    :param k: the highest degree of the Legendre polynomials, so that the resulting Vandermonde matrix
              will have dimension M,n where n = (k+1)*(k+2)/2
    :param M: the number of samples
    :param dt: the integration time step for the ODE
    :param T: the final time of integration for the ODE
    :param Q_ref: the reference value for the average of Q
    :param alpha: a parameter for the confidence interval
    :return: the value of the MCLS estimator, the true error with respect to Q_ref
             and the half-size of the (1-alpha) confidence interval
    """
    # get the Vandermonde matrix, the samples and the number of degrees of freedom
    V, X, n = basis_2d_legendre(k, M, QMC)
    # compute Q for each sample
    Q_val = np.array([QoI(X[0, i], X[1, i], dt, T) for i in range(M)]).T
    # solve the normal equations to find the optimal coefficients for the regression
    if X.shape[1] <= M:
        # if V has full column rank calculate the reduced QR factorization of V ---> reduced to solving a
        # triangular system
        Q, R = np.linalg.qr(V)
        c = np.linalg.solve(R, Q.T @ Q_val)
    else:
        # if V doesn't have full column rank call the solver
        c = np.linalg.solve(V.T @ V, V.T @ Q_val)
    # compute the estimator
    estimate_ = c[0]
    # compute the error with respect to the reference value
    true_error = np.abs(Q_ref - estimate_)
    # compute the estimated error
    estimated_error = 1 / (M-n) * np.linalg.norm(Q_val - V @ c, 2) ** 2
    # compute the half-size of the 1-alpha confidence interval
    confidence_interval = st.norm.ppf(1-alpha/2) * np.sqrt(estimated_error / M)
    return estimate_, true_error, confidence_interval
