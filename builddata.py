import numpy as np
from case1_filters import simulate_ctrv  # 请替换成你实际的导入路径

def simulate_ctrv_markov(

    x0=np.array([0., 0., 10., 0.3, 0.05]),
         dt=0.1,T=1000,# 初始状态
    rho_a=0.95, rho_alpha=0.95,  # AR(1) 系数
    sigma_a_ss=0.5, sigma_alpha_ss=0.1,  # 纵向/角加速度稳态标准差
    sigma_r=0.5, sigma_phi=0.05,  # 测量噪声 std
    seed=None,
    normalize_angle=True
):
    """
    生成：
      xs : (T,5) 状态序列，每行 [x, y, v, theta, omega]
      zs : (T,2) 观测序列，每行 [r, phi]
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # 按给定的稳态标准差计算 eps 的标准差
    sigma_eps_a = sigma_a_ss * np.sqrt(1.0 - rho_a ** 2)
    sigma_eps_alpha = sigma_alpha_ss * np.sqrt(1.0 - rho_alpha ** 2)

    # 初始化马尔科夫噪声
    a_long = np.zeros(T)
    a_alpha = np.zeros(T)
    a_long[0] = rng.normal(0.0, sigma_a_ss)
    a_alpha[0] = rng.normal(0.0, sigma_alpha_ss)

    # 生成噪声序列
    for t in range(1, T):
        a_long[t] = rho_a * a_long[t - 1] + rng.normal(0.0, sigma_eps_a)
        a_alpha[t] = rho_alpha * a_alpha[t - 1] + rng.normal(0.0, sigma_eps_alpha)

    # CTRV 状态轨迹
    xs = np.zeros((T, 5))
    xs[0] = x0.astype(float)

    eps_w = 1e-5
    for t in range(1, T):
        px, py, v, theta, omega = xs[t - 1]
        al, aa = a_long[t - 1], a_alpha[t - 1]

        if abs(omega) > eps_w:
            px_next = px + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            py_next = py - (v / omega) * (np.cos(theta + omega * dt) - np.cos(theta))
            v_next = v + al * dt
            th_next = theta + omega * dt + 0.5 * aa * dt ** 2
            om_next = omega + aa * dt
        else:  # 近似直线
            px_next = px + v * np.cos(theta) * dt
            py_next = py + v * np.sin(theta) * dt
            v_next = v + al * dt
            th_next = theta + 0.5 * aa * dt ** 2
            om_next = aa * dt

        if normalize_angle:
            th_next = np.arctan2(np.sin(th_next), np.cos(th_next))

        xs[t] = [px_next, py_next, v_next, th_next, om_next]

    # 观测序列：极坐标 + 噪声
    r = np.sqrt(xs[:, 0] ** 2 + xs[:, 1] ** 2) + rng.normal(0.0, sigma_r, size=T)
    phi = np.arctan2(xs[:, 1], xs[:, 0]) + rng.normal(0.0, sigma_phi, size=T)
    zs = np.stack([r, phi], axis=1)

    return xs, zs

# 超参数
N  = 1000    # 轨迹条数
T  = 1000    # 每条轨迹长度
dt = 0.1

# 噪声参数
sigma_a      = np.sqrt(0.01)
sigma_alpha  = np.sqrt(0.001)
sigma_r      = np.sqrt(0.5)
sigma_phi    = np.sqrt(0.05)

# 预分配
xs = np.zeros((N, T, 5), dtype=np.float32)
zs = np.zeros((N, T, 2), dtype=np.float32)

for i in range(N):
    # simulate_ctrv 返回 (T,5) 的状态序列 和 (T,2) 的极坐标测量 (r,φ)
    x_seq,z_polar = simulate_ctrv(
        x0=np.array([0., 0., 2., np.pi/4, 0.1]),dt = dt,
        N = T-1,sigma_a = sigma_a,sigma_alpha=sigma_alpha,
        sigma_r   = sigma_r,
        sigma_phi   = sigma_phi
    )
    # 把极坐标转成 x,y
    #r, phi = z_polar[:,0], z_polar[:,1]
    #z_xy   = np.stack([r*np.cos(phi), r*np.sin(phi)], axis=1)  # (T,2)

    xs[i] = x_seq
    zs[i] = z_polar

# 保存成一个 npz 文件
filename = 'vae_dataset.npz'
np.savez(filename, xs=xs, zs=zs)
print("Saved dataset to "+filename)