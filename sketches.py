# Source: https://github.com/lessketching/newtonsketch.git

import torch
import numpy as np

from scipy.linalg import hadamard as hadam_scipy

from time import time

torch.set_default_dtype(torch.float64)

# Gongyu's implementation of leverage score sampling (as sketch)
def sampling(s, probs, data):
  n, d = data.shape
  idx = np.random.choice(n, size=s, p=probs)
  weights = np.tile(probs[idx] ** -0.5, reps=(d, 1))/ np.sqrt(s)
  subsampled = data[idx,:] * weights.T
  return subsampled
  # S = np.zeros((s, n))
  # S[np.arange(s), idx] = 1.
  # # print(f'S has shape {S.shape}')
  # return S

def sqrn_sampling(matrix, s, ):
  # n, d = matrix.shape
  probs = np.linalg.norm(matrix, axis=1) ** 2 / np.linalg.norm(matrix) ** 2
  return sampling(s, probs, matrix)

def lvrg_sampling(matrix, s, ):
  # rank = np.linalg.matrix_rank(data)
  lev_scores = lev_approx(matrix, alpha=5)
  probs = lev_scores / lev_scores.sum()
  return sampling(s, probs, matrix)

###

def _hadamard(matrix):
    n = matrix.shape[0]
    if n == 1:
        return matrix
    _t1 = _hadamard(matrix[:n // 2, ::] + matrix[n // 2:, ::])
    _t2 = _hadamard(matrix[:n // 2, ::] - matrix[n // 2:, ::])
    return torch.cat((_t1, _t2), 0)


def hadamard(matrix):
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1, 1))
    n = matrix.shape[0]
    if n & (n - 1) != 0:
        new_dim = 2 ** (int(np.ceil(np.log(n) / np.log(2))))
        pad_matrix = torch.zeros(new_dim - n, matrix.shape[1]).to(matrix.device)
        matrix = torch.cat((matrix, pad_matrix))
    n = matrix.shape[0]
    diag = np.random.choice([-1, 1], n, replace=True).reshape((-1, 1))
    matrix = torch.Tensor(diag).to(matrix.device) * matrix
    return 1. / np.sqrt(n) * _hadamard(matrix)


def rrs(matrix, sketch_size, nnz=None):
    matrix = torch.from_numpy(matrix)
    n = matrix.shape[0]
    return matrix[np.random.choice(np.arange(n), sketch_size, replace=False), ::]


def rrs_lev_scores(matrix, sketch_size, nnz=None):
    matrix = torch.from_numpy(matrix)
    n, d = matrix.shape
    lev_scores = lev_approx(matrix, alpha=5)
    prob = lev_scores / lev_scores.sum()
    return matrix[np.random.choice(n, sketch_size, replace=False, p=prob), ::]


def gaussian(matrix, sketch_size, nnz=None):
    matrix = torch.from_numpy(matrix)
    S = 1. / np.sqrt(sketch_size) * torch.randn(sketch_size, matrix.shape[0]).to(matrix.device)
    return (S @ matrix).numpy()


def sjlt(matrix, sketch_size, nnz=None):
    # matrix = torch.from_numpy(matrix)
    n, d = matrix.shape
    indices = np.vstack([np.random.choice(sketch_size, n).reshape((1, -1)), np.arange(n)])
    values = np.random.choice(np.array([-1, 1], dtype=np.float64), size=n)
    S = torch.sparse_coo_tensor(indices, values, (sketch_size, n)).to(matrix.device)
    sa = S @ matrix
    return sa


def sparse_rademacher(matrix, sketch_size, nnz=None):
    matrix = torch.from_numpy(matrix)
    n, d = matrix.shape
    if nnz is None:
        nnz = d / n
    d_tilde = int(nnz * n)
    indices = np.vstack([np.repeat(np.arange(sketch_size), d_tilde).reshape((1, -1)),
                         np.random.choice(n, size=sketch_size * d_tilde).reshape((1, -1))])
    values = np.random.choice(np.array([-1, 1], dtype=np.float64), size=sketch_size * d_tilde)
    S = torch.sparse_coo_tensor(indices, values, (sketch_size, n)).to(matrix.device)
    return (np.sqrt(n / (sketch_size * nnz * n)) * S @ matrix).numpy()


def less(matrix, sketch_size, lev_scores=False, nnz=None):
    matrix = torch.from_numpy(matrix)
    if not lev_scores:
        return sparse_rademacher(hadamard(matrix), sketch_size, nnz)
    else:
        n, d = matrix.shape
        lev_scores = lev_approx(matrix, alpha=5)
        prob = lev_scores / lev_scores.sum()
        samples = torch.tensor(np.random.multinomial(d, pvals=prob, size=sketch_size)).to(matrix.device)
        samples = samples / (d * prob.reshape((1, -1)))
        S = torch.sqrt(samples / sketch_size) * torch.tensor(np.random.choice([-1, 1], size=(sketch_size, n))).to(
            matrix.device)
        return (S @ matrix).numpy()


def _srht(indices, v):
    n = v.shape[0]
    if n == 1:
        return v
    i1 = indices[indices < n // 2]
    i2 = indices[indices >= n // 2]
    if len(i1) == 0:
        return _srht(i2 - n // 2, v[:n // 2, ::] - v[n // 2:, ::])
    elif len(i2) == 0:
        return _srht(i1, v[:n // 2, ::] + v[n // 2:, ::])
    else:
        return torch.cat(
            [_srht(i1, v[:n // 2, ::] + v[n // 2:, ::]), _srht(i2 - n // 2, v[:n // 2, ::] - v[n // 2:, ::])], axis=0)


def srht(matrix, sketch_size, nnz=None):
    # matrix = torch.from_numpy(matrix)
    # device = matrix.device
    # matrix = matrix.cpu().numpy()
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1, 1))
    # pad matrix with 0 if first dimension is not a power of 2
    n = matrix.shape[0]
    if n & (n - 1) != 0:
        new_dim = 2 ** (int(np.log(n) / np.log(2)) + 1)
        matrix = torch.cat([matrix, torch.zeros(new_dim - n, matrix.shape[1]).to(matrix.device)], axis=0)
    n = matrix.shape[0]
    indices = np.sort(np.random.choice(np.arange(n), sketch_size, replace=False))
    v = torch.tensor(np.random.choice([-1, 1], n, replace=True)).reshape((-1, 1)).to(matrix.device)
    matrix = v * matrix
    sa = _srht(indices, matrix)
    return sa


def lev_approx(matrix, alpha=10):
    matrix = torch.from_numpy(matrix)
    n, d = matrix.shape
    m = int(alpha * d)
    sa = sjlt(matrix, m)
    _, sig_vec, v_mat = torch.svd(sa)
    y_mat = matrix @ v_mat.T / sig_vec.reshape((1, -1))
    lev_vec = torch.sum(y_mat ** 2, axis=1)
    return lev_vec.cpu().numpy()


if __name__ == "__main__":
    # print(sparse_rademacher(np.random.rand(10, 4), 2).shape)
    from sklearn.datasets import load_svmlight_file
    import matplotlib.pyplot as plt
    X, Y = load_svmlight_file('data/a9a')
    X = X.toarray()
    lev_scores = lev_approx(X)
    idx = np.logical_or((lev_scores > np.percentile(lev_scores, 98)), (lev_scores < np.percentile(lev_scores, 50)))
    large = 75
    small = 30
    lev_large = sum((lev_scores > np.percentile(lev_scores, large)))
    lev_small = sum((lev_scores < np.percentile(lev_scores, small)))
    # print()
    # print(sum((lev_scores > np.percentile(lev_scores, small))))
    print(lev_large / (lev_large + lev_small))
    print(lev_large / X.shape[0])
    # print(sum(np.logical_or((lev_scores > np.percentile(lev_scores, 80)), (lev_scores < np.percentile(lev_scores, 25)))))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(np.compress(idx, lev_scores, axis=0))
    plt.show()
    # idx =
    # test_X =np.compress(bool_idx_Hessian,X,axis=0)
        #     _Y=np.compress(bool_idx_Hessian,Y,axis=0)


'''
def _srht(indices, v):
    n = v.shape[0]
    if n == 1:
        return v
    i1 = indices[indices < n//2]
    i2 = indices[indices >= n//2]
    if len(i1) == 0:
        return _srht(i2-n//2, v[:n//2,::]-v[n//2:,::])
    elif len(i2) == 0:
        return _srht(i1, v[:n//2,::]+v[n//2,::])
    else:
        _t1 = _srht(i1, v[:n//2,::]+v[n//2,::])
        _t2 = _srht(i2-n//2, v[:n//2,::]-v[n//2,::])
        return torch.cat((_t1, _t2), 0)


def srht(matrix, sketch_size, nnz=None):
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1,1))
    n = matrix.shape[0]
    if n & (n-1) != 0:
        new_dim = 2**(int(np.ceil(np.log(n)/np.log(2))))
        pad_matrix = torch.zeros(new_dim-n, matrix.shape[1]).to(matrix.device)
        matrix = torch.cat((matrix, pad_matrix))
    n = matrix.shape[0]
    diag = np.random.choice([-1,1], n, replace=True).reshape((-1,1))
    matrix = torch.Tensor(diag).to(matrix.device) * matrix
    indices = np.sort(np.random.choice(np.arange(n), sketch_size, replace=False))
    return 1./np.sqrt(sketch_size) * _srht(indices, matrix)
    #return _srht(indices, matrix)
'''



