import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from case1_filters import *
from metrics import *
from vae_planar_flow import *

def radar_meas_torch(x):  # x: (..., 5) -> z: (..., 2)
    px = x[..., 0]
    py = x[..., 1]
    r   = torch.sqrt(px * px + py * py + 1e-12)
    phi = torch.atan2(py, px)
    return torch.stack([r, phi], dim=-1)

# --- 角度归一化到 (-pi, pi] ---
def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

# --- 多元高斯的 log-pdf（批量、数值稳定） ---
def mvn_logpdf(diff, R):  # diff: (S,2) or (B,S,2); R: (2,2) (同设备同dtype)
    # 计算 -0.5 * [k*log(2π) + 2*sum(log(diag(L))) + ||L^{-1} diff||^2]
    k = diff.shape[-1]
    L = torch.linalg.cholesky(R)                 # (2,2)
    # 解 L y = diff  -> y = L^{-1} diff
    y = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)  # same shape as diff
    quad = (diff * y).sum(dim=-1)               # (...,)
    logdet = 2.0 * torch.log(torch.diag(L)).sum()
    return -0.5 * (k * math.log(2 * math.pi) + logdet + quad)

def predict_trajectory_fuse(
        model_path,
        zs,
        x0,
        StepVAE,
        R,
        S= 32,
        device=None,
        N=None):
    """
    使用训练好的 StepVAE 模型进行轨迹预测

    参数:
        model_path: str
            已训练模型的文件路径
        zs: np.ndarray, shape (T, 2)
            测量值 (极坐标: r, phi)
        x0: np.ndarray, shape (5,)
            初始状态
        StepVAE: nn.Module
            模型类（定义好的 StepVAE）
        device: torch.device
            torch 设备
        N: int
            预测步数，默认为 zs 的长度

    返回:
        preds: np.ndarray, shape (N, 5)
            预测轨迹
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if N is None:
        N = zs.shape[0]



    # 加载模型
    model = StepVAE().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 极坐标转直角坐标
    r   = zs[:, 0]
    phi = zs[:, 1]
    x_meas = r * np.cos(phi)
    y_meas = r * np.sin(phi)
    zs_xy = np.stack([x_meas, y_meas], axis=1)  # (T,2)
    meas_seq = torch.tensor(zs_xy, dtype=torch.float32).to(device)
    #meas_seq = torch.tensor(zs, dtype=torch.float32, device=device)  # (T,2)
    meas_xy = torch.tensor(np.stack([x_meas, y_meas], axis=1),
                           dtype=torch.float32, device=device)  # (T,2)
    meas_polar = torch.tensor(zs, dtype=torch.float32, device=device)  # (T,2)

    R_t = torch.as_tensor(R, dtype=torch.float32, device=device)       # (2,2)

    preds = torch.zeros((N, 5), device=device)
    prev_x = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)  # (1,5)
    preds[0] = prev_x.squeeze(0)

    with torch.no_grad():
        for k in range(N):
            # —— 模型输入：笛卡尔量测 ——
            meas_xy_k = meas_xy[k].unsqueeze(0)          # (1,2)
            # —— 似然与融合：使用极坐标 ——
            z_k_polar = meas_polar[k]                    # (2,)

            # 采样 S 个候选状态
            x_candidates = []
            for _ in range(S):
                # 若 forward 支持 temperature: model(prev_x, meas_xy_k, temperature=temperature)
                x_pred, mu, logv, _ = model(prev_x, meas_xy_k)
                x_candidates.append(x_pred.squeeze(0))   # (5,)
            X = torch.stack(x_candidates, dim=0)         # (S,5)

            # 极坐标似然 -> log-weights
            z_pred = radar_meas_torch(X)                 # (S,2) -> [r_i, phi_i]
            diff = z_k_polar.unsqueeze(0) - z_pred       # (S,2)
            diff[:, 1] = wrap_angle(diff[:, 1])
            logw = mvn_logpdf(diff, R_t)                 # (S,)
            w = torch.softmax(logw, dim=0)               # (S,)

            # ==== 极坐标融合 ====
            r_i, phi_i = z_pred[:, 0], z_pred[:, 1]
            # 半径加权均值
            r_bar = (w * r_i).sum()
            # 角度圆均值
            c = (w * torch.cos(phi_i)).sum()
            s = (w * torch.sin(phi_i)).sum()
            phi_bar = torch.atan2(s, c)

            # 还原到 (x, y)
            x_bar = r_bar * torch.cos(phi_bar)
            y_bar = r_bar * torch.sin(phi_bar)

            # 其它维度
            v_bar = (w * X[:, 2]).sum()
            # theta 用圆均值
            c_th = (w * torch.cos(X[:, 3])).sum()
            s_th = (w * torch.sin(X[:, 3])).sum()
            theta_bar = torch.atan2(s_th, c_th)
            omega_bar = (w * X[:, 4]).sum()

            x_fused = torch.stack([x_bar, y_bar, v_bar, theta_bar, omega_bar]).unsqueeze(0)  # (1,5)

            preds[k] = x_fused.squeeze(0)  # 存笛卡尔结果
            prev_x = x_fused               # 下一步的 prev_x 也用笛卡尔

    return preds.cpu().numpy()

def predict_trajectory(model_path, zs, x0, StepVAE, device=None, N=None):
    """
    使用训练好的 StepVAE 模型进行轨迹预测

    参数:
        model_path: str
            已训练模型的文件路径
        zs: np.ndarray, shape (T, 2)
            测量值 (极坐标: r, phi)
        x0: np.ndarray, shape (5,)
            初始状态
        StepVAE: nn.Module
            模型类（定义好的 StepVAE）
        device: torch.device
            torch 设备
        N: int
            预测步数，默认为 zs 的长度

    返回:
        preds: np.ndarray, shape (N, 5)
            预测轨迹
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if N is None:
        N = zs.shape[0]

    # 加载模型
    model = StepVAE().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 极坐标转直角坐标
    r   = zs[:, 0]
    phi = zs[:, 1]
    x_meas = r * np.cos(phi)
    y_meas = r * np.sin(phi)
    zs_xy = np.stack([x_meas, y_meas], axis=1)  # (T,2)
    meas_seq = torch.tensor(zs_xy, dtype=torch.float32).to(device)

    # 初始化预测数组
    preds = np.zeros((N, 5), dtype=np.float32)
    preds[0] = x0
    prev_x = torch.tensor(x0, dtype=torch.float32).unsqueeze(0).to(device)  # (1,5)

    with torch.no_grad():
        for k in range(N):
            z_k = meas_seq[k].unsqueeze(0)    # (1,2)
            x_pred, mu, logv, _ = model(prev_x, z_k)  # x_pred: (1,5)
            preds[k] = x_pred.squeeze(0).cpu().numpy()
            prev_x = x_pred

        print("latent std mean:", (F.softplus(logv) + 1e-3).mean().item())
        print("logv:", logv)

    return preds

# 4. 参数设置（对应题中 Q = diag([0.1,0.1,0.01,0.01,0.001]), R = diag([0.5,0.05])）
x0          = np.array([0., 0., 2., np.pi/4, 0.1])
dt, N       = 0.1, 1000
# 过程噪声：这里把加速度噪声 σ_a、角加速度噪声 σ_α 设为 Q 对应项的平方根
sigma_a     = np.sqrt(0.01)    # 对应 v 的过程噪声项
sigma_alpha = np.sqrt(0.001)   # 对应 ω 的过程噪声项
# 测量噪声
sigma_r     = np.sqrt(0.5)
sigma_phi   = np.sqrt(0.05)


# 5. 运行仿真
#xs, zs = simulate_ctrv(x0, dt, N,
                       #sigma_a, sigma_alpha,
                       #sigma_r, sigma_phi)

data = np.load("vae_dataset_markov_noise.npz")

# 假设里面存了 xs 和 zs
xs_all = data["xs"]   # (N, T, 5)
zs_all = data["zs"]
idx =1# np.random.randint(0, N)  # 随机挑一个 0 ~ N-1 的索引
xs= xs_all[idx]  # (T, 5)
zs = zs_all[idx]  # (T, 2)

print("chosen idx:", idx)


P0 = np.eye(5)
#Q  = np.diag([0.1, 0.1, 0.01, 0.01, 0.001])
R  = np.diag([0.5, 0.05])

xs_ekf = ekf_tracking(x0, P0, sigma_a,sigma_alpha, R, zs, dt)
xs_ukf = ukf_tracking(x0, P0, sigma_a,sigma_alpha, R, zs, dt)
#xs_pf = pf_tracking(zs,x0,P0,500,dt, sigma_a, sigma_alpha, R)

name1 = 'step_vae_whitenoise.pth'
name2 = 'step_vae_whitenoise.pth'

preds    = predict_trajectory_fuse(model_path=name1,zs=zs,x0=x0,StepVAE= StepVAE,R=torch.tensor(R, dtype=torch.float32))

preds2    = predict_trajectory(model_path=name2,zs=zs,x0=x0,StepVAE= StepVAE)


rmse = rmse_scalar(xs[:,:2], preds[:,:2])
print(name1+f"全轨迹位置 RMSE: {rmse:.4f}")
rmse2 = rmse_scalar(xs[:,:2], preds2[:,:2])
print(name2+f"全轨迹位置 RMSE: {rmse2:.4f}")

rmse_total_ekf = rmse_scalar(xs[:,:2], xs_ekf[:,:2])
rmse_total_ukf = rmse_scalar(xs[:,:2], xs_ukf[:,:2])
#rmse_total_pf = rmse_scalar(xs[:,:2], xs_pf[:,:2])
print("scalar RMSE of position: EKF: ",rmse_total_ekf,"UKF: ",rmse_total_ukf)#,"PF: ",rmse_total_pf)


# 6. 可视化
plt.figure(figsize=(6,6))
plt.plot(xs[:,0], xs[:,1], '-k', label='True trajectory')
xy_meas = np.vstack([zs[:,0]*np.cos(zs[:,1]),
                     zs[:,0]*np.sin(zs[:,1])]).T
plt.scatter(xy_meas[:,0], xy_meas[:,1],s=1, c='r', label='Measurements')

#plt.plot(xs_ekf[:,0], xs_ekf[:,1], '--b', label='EKF')

#plt.plot(xs_pf[:,0], xs_pf[:,1], '--y', label='PF')
plt.scatter(preds[:,0], preds[:,1], 1,'b', label=name1)
plt.scatter(preds2[:,0], preds2[:,1], 1,'g', label=name2)
#print(preds[:,0])
plt.legend(); plt.axis('equal')
plt.xlabel('x'); plt.ylabel('y')
plt.title('CTRV Simulation with Process & Measurement Noise')
plt.grid(True)
plt.show()
