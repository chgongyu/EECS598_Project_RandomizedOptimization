#######################################
### Subsampled Cubic Regularization ###
#######################################

# Authors: Jonas Kohler and Aurelien Lucchi, 2017

from math import sqrt, ceil, log, isnan
from datetime import datetime
import numpy as np
from sketches import gaussian, srht, less, sparse_rademacher, rrs
from sqrt_hessian import logis_sketched_hessian_sqrt
import torch


def line_search(w, v, g, loss, X, Y, a=0.1, b=0.5):
    delta = (v * g).sum()
    loss_x = loss(w, X, Y)
    s = 1
    ws = w + s * v
    while loss(ws, X, Y,) > loss_x + a * s * delta:
        s = b * s
        ws = w + s * v
    return s

# def line_search(x, dx, dg, g, X, Y, a=0.1, b=0.5):
#     """Perform backtracking line search.
#     Backtracking line search begins with an initial step-size dx and backtracks
#     until the adjusted linear estimate overestimates the loss function $g$.
#     For more information refer to pgs. 464-466 of Convex Optimization by Boyd.
#     Args:
#         x (np.ndarray): Coefficients
#         dx (np.ndarray): Step direction
#         g (function): Loss function
#         dg (function): Loss function gradient
#         a (numeric): scaling factor
#         b (numeric): reduction factor
#     Returns:
#         float
#     """
#     mu = 1
#     while (g(x, X, Y) + mu * a * dg.T.dot(dx)
#             < g(x + mu * dx, X, Y)):
#         mu = mu * b
#     return mu

def gen_sketch_mat(m, n, method):
    """Generate a sketch matrix.
    A sketch matrix $S\in\mathbb{R}^{m\times n}$ has the property that
    $\mathbb{E} S^T S = \mathbb{I}/m$.
    Args:
        m (int): number of rows of the sketch matrix (desired rank)
        n (int): number of columns of the sketch matrix (size of matrix
            to be sketched)
        method (str): method for generating the sketch matrix.  Currently,
            only random normal sketch matrices are supported.
    Returns:
        np.ndarray: a sketch matrix
    """
    if method is 'Gaussian':
        S = np.random.randn(m, n) / m
    elif method is 'Rademacher':
        # Produces r.v. in {0, 1} with equal probability
        S = ((np.random.randn(m, n) > 0) * 2 - 1) / m
    else:
        raise ValueError('Unrecognized sketch type: ' + method)
    return S

def NS(w, loss, gradient, Hv=None, hessian=None, X=None, Y=None, opt=None, **kwargs):
    """
    Minimize a continous, unconstrained function using the Sketched Newton.

    References
    ----------
    Cartis, C., Gould, N. I., & Toint, P. L. (2011). Adaptive cubic regularisation methods for unconstrained optimization. Part I: motivation, convergence and numerical results. Mathematical Programming, 127(2), 245-295.
    Chicago

    Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods. Society for Industrial and Applied Mathematics.

    Kohler, J. M., & Lucchi, A. (2017). Sub-sampled Cubic Regularization for Non-convex Optimization. arXiv preprint arXiv:1705.05933.


    Parameters
    ----------
    loss : callable f(x,**kwargs)
        Objective function to be minimized.

    grad : callable f'(x,**kwargs), optional
        Gradient of f.
    Hv: callable Hv(x,**kwargs), optional
        Matrix-vector-product of Hessian of f and arbitrary vector v
    **kwargs : dict, optional
        Extra arguments passed to loss, gradient and Hessian-vector-product computation, e.g. regularization constant or number of classes for softmax regression.
    opt : dictionary, optional
        optional arguments passed to ARC
    """
    print('--- Newton Sketch ---\n')

    ### Set Parameters ###

    if X is None:
        n = 1
        d = 1
    else:
        n = X.shape[0]
        d = X.shape[1]

        # Basics
    sketch_size = opt.get('sketch_size', int(np.sqrt(X.shape[0])))
    sketch_type = opt.get('sketch_type', sparse_rademacher)
    alpha = opt.get('alpha', 1e-3)

    grad_tol = opt.get('grad_tol', 1e-6)
    n_iterations = opt.get('n_iterations', 100)

    ### -> no opt call after here!!
    k = 0
    n_samples_seen = 0

    grad = gradient(w, X, Y, **kwargs)
    grad_norm = np.linalg.norm(grad)

    loss_collector = []
    timings_collector = []
    samples_collector = []

    _loss = loss(w, X, Y, **kwargs)
    loss_collector.append(_loss)
    timings_collector.append(0)
    samples_collector.append(0)

    start = datetime.now()
    timing = 0

    for i in range(n_iterations):

        #### I: Sketching #####
        B = logis_sketched_hessian_sqrt(X, w)

        sqrt_hessian = sketch_type(B, sketch_size)

        # S = gen_sketch_mat(sketch_size, n, 'Gaussian')
        # S = np.eye(n)
        # sqrt_hessian = S @ B

        #### II: Step computation #####
        # a) recompute gradient either because of accepted step or because of re-sampling
        grad = gradient(w, X, Y, **kwargs)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < grad_tol:
            break

        # b) call subproblem solver
        H = sqrt_hessian.T @ sqrt_hessian + alpha*np.eye(d)
        # hessian = H + opt['alpha'] * np.eye(d)
        # H = hessian(w, X, Y, **kwargs)

        # print(f'{np.linalg.norm(H+ alpha * np.eye(d) - H)}')
        v = - np.linalg.solve(H, grad)
        mu = line_search(w, v, grad, loss, X, Y)

        s = mu * v
        sn = np.linalg.norm(s)

        #### III: Regularization Update #####
        # previous_f = loss(w, X, Y, **kwargs)
        w = w + s
        # w = w + 0.01*v
        current_f = loss(w, X, Y, **kwargs)
        _loss = current_f


        ### IV: Save Iteration Information  ###
        _timing = timing
        timing = (datetime.now() - start).total_seconds()
        print('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(
            grad_norm), 'time= ', round(timing - _timing, 3),
              # 'penalty=', sigma,
              'stepnorm=', sn,
              # 'Samples Hessian=',
              # sample_size_Hessian, 'samples Gradient=', sample_size_gradient, "\n"
              )

        timings_collector.append(timing)
        samples_collector.append(n_samples_seen)

        loss_collector.append(_loss)

        k += 1
    return w, timings_collector, loss_collector, samples_collector


