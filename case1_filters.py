import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.monte_carlo import systematic_resample
from metrics import rmse,rmse_scalar,compute_nees




def ctrv_dynamics(x, dt, sigma_a, sigma_alpha):
    """严格正确的CTRV离散化模型（含过程噪声）"""
    v, theta, omega = x[2], x[3], x[4]

    # 生成过程噪声
    a_long = np.random.normal(sigma_a, sigma_a)+np.random.normal(-sigma_a, sigma_a)
    a_ang = np.random.normal(sigma_alpha, sigma_alpha)+np.random.normal(-sigma_alpha, sigma_alpha)

    # 状态更新
    if np.abs(omega) > 1e-5:
        x_next = np.array([
            x[0] + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta)),

            x[1] - (v / omega) * (np.cos(theta + omega * dt) - np.cos(theta)),
            v + a_long * dt,
            theta + omega * dt + 0.5 * a_ang * dt ** 2,
            omega + a_ang * dt
        ])
    else:  # 直线运动
        x_next = np.array([
            x[0] + v * np.cos(theta) * dt ,
            x[1] + v * np.sin(theta) * dt ,
            v + a_long * dt,
            theta + 0.5 * a_ang * dt ** 2,
            a_ang * dt
        ])
    return x_next

def radar_meas(x):
    px, py = x[0], x[1]
    r   = np.hypot(px, py)
    phi = np.arctan2(py, px)
    return np.array([r, phi])

# 3. 仿真函数
def simulate_ctrv(x0, dt, N,
                  sigma_a, sigma_alpha,
                  sigma_r, sigma_phi):
    """
    x0            初始状态 [x,y,v,θ,ω]
    dt, N         时间步长和步数
    sigma_*       过程/测量噪声各分量的标准差
    """
    xs = np.zeros((N+1, 5))
    zs = np.zeros((N+1, 2))
    xs[0] = x0
    # 初始也可打量测噪声
    zs[0] = radar_meas(x0) + np.random.normal(0, [sigma_r, sigma_phi])

    for k in range(N):
        # 1) 状态传播（带过程噪声）
        xs[k+1] = ctrv_dynamics(xs[k], dt, sigma_a, sigma_alpha)
        # 2) 生成观测（带测量噪声）
        z_true   = radar_meas(xs[k+1])
        zs[k+1]  = z_true + np.random.normal(0, [sigma_r, sigma_phi])

    return xs, zs


def compute_process_noise_Q(theta, dt, sigma_a, sigma_alpha):
    """
    Compute the discrete-time process noise covariance Q for the CTRV model
    using Bar‐Shalom’s formula, given current heading theta, time step dt,
    longitudinal accel noise std sigma_a, angular accel noise std sigma_alpha.
    """
    ca, sa = np.cos(theta), np.sin(theta)
    dt2, dt3, dt4 = dt**2, dt**3, dt**4
    Q = np.zeros((5,5))
    # Position‐velocity block (x,y,v)
    Q[0,0] = dt4/4 * sigma_a**2 * ca*ca
    Q[0,1] = dt4/4 * sigma_a**2 * sa*ca
    Q[1,0] = Q[0,1]
    Q[1,1] = dt4/4 * sigma_a**2 * sa*sa

    Q[0,2] = dt3/2 * sigma_a**2 * ca
    Q[2,0] = Q[0,2]
    Q[1,2] = dt3/2 * sigma_a**2 * sa
    Q[2,1] = Q[1,2]
    Q[2,2] = dt2   * sigma_a**2

    # Heading‐turnrate block (theta,omega)
    Q[3,3] = dt4/4 * sigma_alpha**2
    Q[3,4] = dt3/2 * sigma_alpha**2
    Q[4,3] = Q[3,4]
    Q[4,4] = dt2   * sigma_alpha**2

    return Q



def ekf_tracking(x0, P0, sigma_a, sigma_alpha,R, zs, dt):
    """
    用 FilterPy 的 ExtendedKalmanFilter.predict_update 完成 EKF
    x0: 初始状态 (5,)
    P0: 初始协方差 (5,5)
    Q , R: 过程噪声和测量噪声协方差
    zs: 测量序列，shape = (N+1, 2)
    dt: 时间步长
    返回 xs_ekf: shape = (N+1, 5)
    """
    ekf = ExtendedKalmanFilter(dim_x=5, dim_z=2)
    ekf.x = x0.copy()
    ekf.P = P0.copy()
    ekf.R = R.copy()

    def fx(x, u):
        return ctrv_dynamics(x, u, 0, 0)

    def F_jacobian(x, dt):
        n, eps = len(x), 1e-5
        F = np.zeros((n,n))
        f0 = fx(x, dt)
        for i in range(n):
            xp = x.copy(); xp[i] += eps
            F[:,i] = (fx(xp, dt) - f0) / eps
        return F

    def H_jacobian(x):
        """带正则项的观测雅可比，避免除零"""
        px, py = x[0], x[1]
        eps = 1e-6
        d2 = max(px*px + py*py, eps)
        d  = np.sqrt(d2)
        H = np.zeros((2,5))
        H[0,0], H[0,1] = px/d,     py/d
        H[1,0], H[1,1] = -py/d2,   px/d2
        return H

    N = len(zs)
    xs_ekf = np.zeros((N,5))
    xs_ekf[0] = x0

    for k in range(1, N):
        theta = ekf.x[3]
        ekf.Q = compute_process_noise_Q(theta, dt, sigma_a, sigma_alpha)
        # 1) 计算当前点的状态转移雅可比，并赋给 ekf.F
        F = F_jacobian(ekf.x, dt)     # 你之前定义的 jacobian_F
        ekf.F = F

        # 2) 用 predict_update 完成 Predict + Update
        ekf.predict_update(
            zs[k],                  # 本次测量
            HJacobian=H_jacobian,   # H 的雅可比函数
            Hx=radar_meas,          # 观测函数
            u=dt                    # 传入 dt 作为控制量，让 predict_update 调用 predict_x(u)
        )

        xs_ekf[k] = ekf.x.copy()

    return xs_ekf





# ———— UKF 跟踪 ————
def ukf_tracking(x0, P0, sigma_a, sigma_alpha, R, zs, dt):
    """
    对测量序列 zs 做 UKF，返回状态估计序列 xs_ukf
    参数同 ekf_tracking
    """
    # sigma 点生成
    points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(
        dim_x=5, dim_z=2, dt=dt,
        fx=lambda x, dt: ctrv_dynamics(x, dt, 0, 0),
        hx=lambda x: radar_meas(x),
        points=points
    )
    ukf.x = x0.copy()
    ukf.P = P0.copy()
    ukf.R = R.copy()

    # 迭代滤波
    N = len(zs)
    xs_ukf = np.zeros((N,5))
    xs_ukf[0] = x0
    for k in range(1, N):
        theta = ukf.x[3]
        ukf.Q = compute_process_noise_Q(theta, dt, sigma_a, sigma_alpha)
        ukf.predict()
        ukf.update(zs[k])
        xs_ukf[k] = ukf.x

    return xs_ukf



def pf_tracking(zs, x0, P0, Np, dt, sigma_a, sigma_alpha, R):
    # 初始化

    def gaussian_pdf(x, mean, cov):
        """
        计算多元高斯概率密度 p(x; mean, cov)
        x:   (k,) 偏差向量
        mean:(k,) 均值，通常为 0
        cov: (k,k) 协方差矩阵
        """
        k = x.size
        dev = x - mean
        inv_cov = np.linalg.inv(cov)
        norm = np.sqrt((2 * np.pi) ** k * np.linalg.det(cov))
        return np.exp(-0.5 * dev.T @ inv_cov @ dev) / norm

    particles = np.random.multivariate_normal(x0, P0, size=Np)
    weights   = np.ones(Np)/Np
    xs_pf     = np.zeros((len(zs), x0.size))

    dim = x0.size

    for k, z in enumerate(zs):
        # —预测：对每个粒子进状态转移—
        for i in range(Np):
            particles[i] = ctrv_dynamics(particles[i], dt, sigma_a, sigma_alpha)

        # —更新权重：用观测似然—
        for i, p in enumerate(particles):
            z_pred = radar_meas(p)
            diff   = z - z_pred
            diff[1] = (diff[1]+np.pi)%(2*np.pi)-np.pi
            weights[i] *= gaussian_pdf(diff, np.zeros(2), R)

        # 正则化
        weights += 1e-300
        weights /= np.sum(weights)

        # —估计—
        xs_pf[k] = np.average(particles, axis=0, weights=weights)

        # —重采样（当有效样本数过低）—
        Neff = 1.0/np.sum(weights**2)
        if Neff < Np/3:
            idx        = systematic_resample(weights)
            particles  = particles[idx]
            weights[:] = 1.0/Np

        xnext = np.average(particles, axis=0, weights=weights)
        xs_pf[k] = xnext


        # —计算加权协方差 P_k —
        Pnext = np.zeros((dim, dim))
        for i in range(Np):
            d = particles[i] - xnext
            # 如果状态中包含角度分量(例如索引3)，请在这里做角度归一化
            # d[3] = (d[3]+np.pi)%(2*np.pi)-np.pi
            Pnext += weights[i] * np.outer(d, d)

        # —用 N(x̂_k, P_k) 重新生成下一步粒子—
        particles = np.random.multivariate_normal(xnext, Pnext, size=Np)
        weights = np.ones(Np) / Np

    return xs_pf