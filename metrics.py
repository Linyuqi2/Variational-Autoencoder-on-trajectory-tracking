import numpy as np

def rmse(y_true, y_pred, axis=0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # squared errors
    se = (y_true - y_pred)**2
    # mean squared error
    mse = np.mean(se, axis=axis)
    # root
    return np.sqrt(mse)

def rmse_scalar(y_true, y_pred):
    return np.sqrt(np.mean(np.sum((y_true - y_pred)**2, axis=1)))

def compute_nees(xs_true, xs_est, Ps):
    """
    计算 NEES 序列。

    参数
    ----
    xs_true : ndarray, shape (N, n)
        真实状态序列。
    xs_est  : ndarray, shape (N, n)
        估计状态序列。
    Ps      : ndarray, shape (N, n, n)
        对应每个时刻的误差协方差矩阵 P_k。

    返回
    ----
    nees : ndarray, shape (N,)
        每个时刻的 NEES 值。
    """
    xs_true = np.asarray(xs_true)
    xs_est  = np.asarray(xs_est)
    Ps       = np.asarray(Ps)
    N, n     = xs_true.shape

    nees = np.zeros(N)
    for k in range(N):
        e = xs_est[k] - xs_true[k]            # 误差向量
        P = Ps[k]
        nees[k] = e.T @ np.linalg.inv(P) @ e  # NEES_k
    return nees