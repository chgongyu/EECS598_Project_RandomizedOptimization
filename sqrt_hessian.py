# Source: https://github.com/huisaddison/newton-sketch.git

import numpy as np
from loss_functions import phi
def safe_sigmoid(x):
    return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))

def logis_laplacian_weights(A, x):
    """Returns the weights of the Hessian of the (negative)
    log-likelihood.

    The Hessian of the negative log-likelihood may be rewritten as:
    $$
        A^T W A
    $$
    where $W$ is a diagonal matrix, with:
    $$
        W_{ii} = p(a_i; x)(1 - p(a_i; x) = \frac{1}{(1 + \exp(a_i^T x))^2}
    $$

    Args:
        A (np.ndarray): Feature matrix
        x (np.ndarray): Coefficients

    Returns:
        np.ndarray: the diagonal of the weight matrix $W$ as defined above.
    """
    e = safe_sigmoid(A@x)
    return e*(1-e)


def logis_sketched_hessian_sqrt(A, x, ):
    """Returns the square root of the sketched Hessian for the negative
    log-likelihood.

    The sketched Hessian takes the form
    $$
        A^T S^T W S A
    $$
    and because of its diagonal form, it may be represented as:
    $$
        B^T B
    $$
    where
    $$
        B = SAW^{\frac{1}{2}}
    $$

    Args:
        A (np.ndarray): Feature matrix
        x (np.ndarray): Coefficients
        y (np.ndarray): Targets
        S (np.ndarray): Sketch matrix

    Returns:
        np.ndarray: square root of Hessian, as described above as $B$.
    """
    # Reshape into column vector to broadcast across rows
    # z = (A @ x).reshape((-1, ))
    # q = phi(z)
    # h = np.array(q * (1 - q))
    # d = (h ** 0.5).reshape((-1, 1))

    # print(h.shape)
    d = (logis_laplacian_weights(A, x, ) ** 0.5).reshape((-1, 1))

    # return S.dot(A * d)
    return A * d / np.sqrt(A.shape[0])


if __name__ == "__main__":
    from sklearn.datasets import load_svmlight_file
    import numpy as np
    import loss_functions

    X, Y = load_svmlight_file('data/a9a')
    X = X.toarray()
    n, d = X.shape
    w = np.random.rand(X.shape[1])
    z = X.dot(w).reshape((-1, ))
    q = phi(z)
    h = np.array(q * (1 - q))
    # np.zeros((4, 4)) * np.zeros((4, 1))

    H1 = logis_sketched_hessian_sqrt(X, w)
    H = np.dot(np.transpose(X), h[:, np.newaxis] * X) / n
    H2 = loss_functions.logistic_loss_hessian(w, X, Y)

    alpha = 1e-3
    reg = alpha * np.eye(d, d)
    print(np.linalg.norm(H1.T @ H1  + reg - H2))