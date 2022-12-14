#######################################
### Subsampled Cubic Regularization ###
#######################################

# Authors: Jonas Kohler and Aurelien Lucchi, 2017

from math import sqrt, ceil, log, isnan
from datetime import datetime
import numpy as np
import time
from sklearn.utils.extmath import row_norms
import scipy
import warnings
import copy

def subsampled_regnewton(w, loss, gradient, Hv=None, hessian=None, X=None, Y=None, opt=None,**kwargs):

    """
    Minimize a continous, unconstrained function using the Adaptive Cubic Regularization method.

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
    print ('--- Subsampled Regularized Newton ---\n')

    ### Set Parameters ###

    if X is None:
        n=1 
        d=1
    else:
        n = X.shape[0] 
        d = X.shape[1] 

    # Basics
    gamma_1 = opt.get('penalty_increase_multiplier',2.)
    gamma_2 = opt.get('penalty_derease_multiplier',2.)
    assert (gamma_1 >= 1 and gamma_2 >= 1), "penalty update parameters must be greater or equal to 1"

    sigma = opt.get('initial_penalty_parameter',1.)
    grad_tol = opt.get('grad_tol',1e-6)
    n_iterations = opt.get('n_iterations',100)   

    # Subproblem
    l2 = opt.get('l2', 1e-3)
    adaptive = opt.get('adaptive_regnewton', False)

    # Sampling
    Hessian_sampling_flag=opt.get('Hessian_sampling', False)
    gradient_sampling_flag=opt.get('gradient_sampling', False)

    if gradient_sampling_flag==True or Hessian_sampling_flag==True:
        assert (X is not None and Y is not None), "Subsampling is only possible if data is passsed, i.e. X and Y may not be none"

    initial_sample_size_Hessian=opt.get('initial_sample_size_Hessian',0.05)
    initial_sample_size_gradient=opt.get('initial_sample_size_gradient',0.05)
    sample_scaling_Hessian=opt.get('sample_scaling_Hessian',1)
    sample_scaling_gradient=opt.get('sample_scaling_gradient',1)
    unsuccessful_sample_scaling=opt.get('unsuccessful_sample_scaling',1.25)
    sampling_scheme=opt.get('sampling_scheme', 'adaptive')
    if Hessian_sampling_flag==False and gradient_sampling_flag ==False:
        sampling_scheme=None
   
    print("- Hessian_sampling:" , Hessian_sampling_flag)
    print("- Gradient_sampling:", gradient_sampling_flag)
    print("- Sampling_scheme:" , sampling_scheme,"\n")

    ### -> no opt call after here!!
    k=0
    n_samples_seen=0

    successful_flag=True

    grad = gradient(w, X, Y,**kwargs)  
    grad_norm=np.linalg.norm(grad)

    loss_collector=[]
    timings_collector=[]
    samples_collector=[]
    
    _loss = loss(w, X, Y, **kwargs) 
    loss_collector.append(_loss)
    timings_collector.append(0)
    samples_collector.append(0)

    start = datetime.now()
    timing=0

    # compute exponential growth constant such that full sample size is reached in n_iterations
    if sampling_scheme=='exponential':
        exp_growth_constant=((1-initial_sample_size_Hessian)*n)**(1/n_iterations)

    # compute Hessian Lipschitz of loss function
    
    grad_old = None
    H = None

    for i in range(n_iterations):
        time0 = datetime.now()

        #### I: Subsampling #####
        ## a) determine batchsize ##
        if sampling_scheme=='exponential':
            sample_size_Hessian = Hessian_sampling_flag*(int(min(n, n*initial_sample_size_Hessian + exp_growth_constant**(i+1)))+1) + (1-Hessian_sampling_flag)*n
            sample_size_gradient= gradient_sampling_flag*(int(min(n, n*initial_sample_size_gradient + exp_growth_constant**(i+1)))+1) + (1-gradient_sampling_flag)*n

        elif sampling_scheme=='linear':
            sample_size_Hessian = Hessian_sampling_flag*int(min(n, max(n*initial_sample_size_Hessian, n/n_iterations*(i+1))))+(1-Hessian_sampling_flag)*n
            sample_size_gradient= gradient_sampling_flag*int(min(n, max(n*initial_sample_size_gradient, n/n_iterations*(i+1))))+(1-gradient_sampling_flag)*n

        elif sampling_scheme=='adaptive':
            if i==0:
                sample_size_Hessian=Hessian_sampling_flag*int(initial_sample_size_Hessian*n)+(1-Hessian_sampling_flag)*n
                sample_size_gradient=gradient_sampling_flag*int(initial_sample_size_gradient*n)+(1-gradient_sampling_flag)*n
            else:
                #adjust sampling constant c such that the first step would have given a sample size of initial_sample_size
                if i==1:
                    c_Hessian=(initial_sample_size_Hessian*n*khgk)/log(d)
                    c_gradient=(initial_sample_size_gradient*n*((sn**2)*khgk))/log(d)
                if successful_flag==False:
                    sample_size_Hessian=Hessian_sampling_flag*min(n,int(sample_size_Hessian*unsuccessful_sample_scaling)) + (1-Hessian_sampling_flag)*n
                    sample_size_gradient=gradient_sampling_flag*min(n,int(sample_size_gradient*unsuccessful_sample_scaling)) +(1-gradient_sampling_flag)*n
                else:
                    sample_size_Hessian=Hessian_sampling_flag*min(n,int(max((c_Hessian*log(d)/sn*sample_scaling_Hessian),initial_sample_size_Hessian*n))) + (1-Hessian_sampling_flag)*n            
                    sample_size_gradient=gradient_sampling_flag*min(n,int(max((c_gradient*log(d)/(((sn**2)*khgk))*sample_scaling_gradient),initial_sample_size_gradient*n))) + (1-gradient_sampling_flag)*n
            # else:
            #     #adjust sampling constant c such that the first step would have given a sample size of initial_sample_size
            #     if i==1:
            #         c_Hessian=(initial_sample_size_Hessian*n*sn**2)/log(d)
            #         c_gradient=(initial_sample_size_gradient*n*sn**4)/log(d)
            #     if successful_flag==False:
            #         sample_size_Hessian=Hessian_sampling_flag*min(n,int(sample_size_Hessian*unsuccessful_sample_scaling)) + (1-Hessian_sampling_flag)*n
            #         sample_size_gradient=gradient_sampling_flag*min(n,int(sample_size_gradient*unsuccessful_sample_scaling)) +(1-gradient_sampling_flag)*n
            #     else:
            #         sample_size_Hessian=Hessian_sampling_flag*min(n,int(max((c_Hessian*log(d)/(sn**2)*sample_scaling_Hessian),initial_sample_size_Hessian*n))) + (1-Hessian_sampling_flag)*n            
            #         sample_size_gradient=gradient_sampling_flag*min(n,int(max((c_gradient*log(d)/(sn**4)*sample_scaling_gradient),initial_sample_size_gradient*n))) + (1-gradient_sampling_flag)*n
        else:
            sample_size_Hessian=n
            sample_size_gradient=n

        ## b) draw batches ##
        if sample_size_Hessian <n:
            int_idx_Hessian=np.random.randint(0, high=n, size=sample_size_Hessian)        

            bool_idx_Hessian = np.zeros(n,dtype=bool)
            bool_idx_Hessian[int_idx_Hessian]=True
            _X=np.zeros((sample_size_Hessian,d))
            _X=np.compress(bool_idx_Hessian,X,axis=0)
            _Y=np.compress(bool_idx_Hessian,Y,axis=0)

        else: 
            _X=X
            _Y=Y

        if sample_size_gradient < n:
            int_idx_gradient=np.random.randint(0, high=n, size=sample_size_gradient)        
            bool_idx_gradient = np.zeros(n,dtype=bool)
            bool_idx_gradient[int_idx_gradient]=True
            _X2=np.zeros((sample_size_gradient,d))
            _X2=np.compress(bool_idx_gradient,X,axis=0)
            _Y2=np.compress(bool_idx_gradient,Y,axis=0)

        else:
            _X2=X
            _Y2=Y

        n_samples_per_step=sample_size_Hessian+sample_size_gradient

        
        #### II: Step computation #####
        # a) recompute gradient either because of accepted step or because of re-sampling
        
        if gradient_sampling_flag==True or successful_flag==True:
            grad = gradient(w, _X2, _Y2, **kwargs)  
            grad_norm =np.linalg.norm(grad)  
            if grad_norm < grad_tol:
                break
        hess = hessian(w, _X, _Y, **kwargs)
        time1 = datetime.now()
        if i == 0:
            hess_lip = hessian_lip_logistic_reg(_X, l2)
            origin_hess_lip = hess_lip
        # b) solve subproblem
        if hess_lip is None:
            hess_lip = 1e-5
        if adaptive and grad_old is not None:
            hess_lip /= 2
            empirical_lip = empirical_hess_lip(grad, grad_old, hess, w, w_old)
            hess_lip = max(hess_lip, empirical_lip)
        if adaptive or H is None:
            H = hess_lip / 2
        time2 = datetime.now()
        grad_norm = np.linalg.norm(grad)
        identity_coef = (H * grad_norm)**0.5
        grad_old = copy.deepcopy(grad)
        w_old = copy.deepcopy(w)
        delta_w = -np.linalg.solve(hess + identity_coef*np.eye(d), grad)
        sn = np.linalg.norm(delta_w)
        w = w + delta_w
        khgk = origin_hess_lip * grad_norm
        time3 = datetime.now()
        ### IV: Save Iteration Information  ###
        _loss = loss(w, X, Y, **kwargs) 
        time4 = datetime.now()
        n_samples_seen += n_samples_per_step
        _timing=timing
        # timing += (time3 - time0).total_seconds()
        timing = (datetime.now() - start).total_seconds()

        print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(
            grad_norm), 'time= ', round(timing-_timing,5), 'penalty=', sigma, 'stepnorm=', sn, 'Samples Hessian=', sample_size_Hessian,'samples Gradient=', sample_size_gradient,'H', H, "\n")
        print('grad and hess time: {}, hess lip time: {}, solver time: {}, loss time: {}'.format((time1 - time0).total_seconds(), (time2 - time1).total_seconds(), (time3 - time2).total_seconds(), (time4 - time3).total_seconds()))
        timings_collector.append(timing)
        samples_collector.append(n_samples_seen)

        loss_collector.append(_loss)

        k += 1

    return w,timings_collector,loss_collector, samples_collector


def empirical_hess_lip(grad, grad_old, hess, x, x_old):
    grad_error = grad - grad_old - hess@(x - x_old)
    r2 = np.linalg.norm(x - x_old)**2
    if r2 > 0:
        return 2 * np.linalg.norm(grad_error) / r2
    return np.finfo(float).eps

############################
### Auxiliary Functions ###
############################
def mitternachtsformel(a,b,c):
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper

def hessian_lip_logistic_reg(A, l2=0):
    a_max = row_norms(A, squared=False).max()
    A_norm = (smoothness(A, l2) - l2) * 4
    _hessian_lipschitz = A_norm * a_max / (6*np.sqrt(3))
    return _hessian_lipschitz

def smoothness(A, l2):
    n, dim = A.shape
    if dim > 20000 and n > 20000:
        warnings.warn("The matrix is too large to estimate the smoothness constant, so Frobenius estimate is used instead.")
        if scipy.sparse.issparse(A):
            _smoothness = 0.25*scipy.sparse.linalg.norm(A, ord='fro')**2/n + l2
        else:
            _smoothness = 0.25*np.linalg.norm(A, ord='fro')**2/n + l2
    else:
        sing_val_max = scipy.sparse.linalg.svds(A, k=1, return_singular_vectors=False)[0]
        _smoothness = 0.25*sing_val_max**2/n + l2
    return _smoothness
