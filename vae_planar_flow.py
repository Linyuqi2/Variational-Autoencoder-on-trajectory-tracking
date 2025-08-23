import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset

import matplotlib.pyplot as plt

from case1_filters import ctrv_dynamics, radar_meas, simulate_ctrv, compute_process_noise_Q
from metrics import rmse_scalar


# --- 1. Data generation: simulate trajectories and measurements ---

def ctrv_dynamics_torch(x, dt, sigma_a, sigma_alpha, *, generator=None):
    """
    严格正确的 CTRV 离散模型（含过程噪声）的 PyTorch 版。
    参数
    ----
    x           : (..., 5) tensor = [px, py, v, theta, omega]
    dt          : float 或 tensor（可广播）
    sigma_a     : 标量/张量，加速度噪声“σ”
    sigma_alpha : 标量/张量，角加速度噪声“σ”
    generator   : 可选 torch.Generator，用于可复现实验

    返回
    ----
    x_next      : (..., 5) tensor
    """

    # 保证类型/设备对齐
    x         = torch.as_tensor(x)
    device    = x.device
    dtype     = x.dtype
    dt        = torch.as_tensor(dt, device=device, dtype=dtype)

    # 噪声：原实现 a ~ N(+σ, σ^2) + N(-σ, σ^2)  =>  等价 N(0, 2σ^2)
    # 这里直接采 N(0, sqrt(2)*σ)
    # 注意：如果 sigma_* 是 Python 标量，会自动广播
    std_long  = math.sqrt(2.0) * float(sigma_a)
    std_ang   = math.sqrt(2.0) * float(sigma_alpha)

    a_long = torch.randn(x.shape[:-1], device=device, dtype=dtype, generator=generator) * std_long
    a_ang  = torch.randn(x.shape[:-1], device=device, dtype=dtype, generator=generator) * std_ang

    px, py, v, theta, omega = x.unbind(dim=-1)

    # 旋转运动与直线运动的分支
    mask_turn = omega.abs() > 1e-5

    # --- 转弯情形 ---
    # Δx = (v/ω) [sin(θ+ωΔt) - sin(θ)]
    # Δy = -(v/ω) [cos(θ+ωΔt) - cos(θ)]
    theta_next_turn = theta + omega * dt + 0.5 * a_ang * dt**2
    px_next_turn = px + (v / omega) * (torch.sin(theta + omega * dt) - torch.sin(theta))
    py_next_turn = py - (v / omega) * (torch.cos(theta + omega * dt) - torch.cos(theta))
    v_next_turn  = v + a_long * dt
    omg_next_turn = omega + a_ang * dt

    # --- 直线情形 ---
    theta_next_line = theta + 0.5 * a_ang * dt**2
    px_next_line = px + v * torch.cos(theta) * dt
    py_next_line = py + v * torch.sin(theta) * dt
    v_next_line  = v + a_long * dt
    omg_next_line = a_ang * dt

    # 按 mask 选择
    px_next   = torch.where(mask_turn, px_next_turn,   px_next_line)
    py_next   = torch.where(mask_turn, py_next_turn,   py_next_line)
    v_next    = torch.where(mask_turn, v_next_turn,    v_next_line)
    theta_next= torch.where(mask_turn, theta_next_turn,theta_next_line)
    omg_next  = torch.where(mask_turn, omg_next_turn,  omg_next_line)

    x_next = torch.stack([px_next, py_next, v_next, theta_next, omg_next], dim=-1)
    return x_next

def build_dataset(n_traj=1000, T=50, dt=0.1):
    sigma_a     = np.sqrt(0.01)
    sigma_alpha = np.sqrt(0.001)
    sigma_r     = np.sqrt(0.5)
    sigma_phi   = np.sqrt(0.05)

    meas_list = []
    true_list = []
    for _ in range(n_traj):
        xs, zs = simulate_ctrv(
            x0=np.array([0.,0.,2.,np.pi/4,0.1]),
            dt=dt, N=T-1,
            sigma_a=sigma_a, sigma_alpha=sigma_alpha,
            sigma_r=sigma_r, sigma_phi=sigma_phi
        )
        # xs: (T,5), zs: (T,2)

          # shape (T,2)
        meas_list.append(zs)
        true_list.append(xs)       # 真值 x_0…x_{T-1}
    # 转成 float32 tensor
    meas_arr = np.array(meas_list, dtype=np.float32)
    true_arr = np.array(true_list, dtype=np.float32)
    return torch.from_numpy(meas_arr), torch.from_numpy(true_arr)

def get_fixed_A_CTRV(x0, dt):
    """
    输入: x0 (numpy array, 长度5), dt 为步长
    输出: torch.Tensor A ∈ ℝ⁵ˣ⁵，表示 CTRV 模型的雅可比
    """
    x, y, v, theta, omega = x0
    if abs(omega) < 1e-4:
        omega = 1e-4  # 避免除0

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_theta_dt = np.sin(theta + omega * dt)
    cos_theta_dt = np.cos(theta + omega * dt)

    A = np.eye(5)
    A[0, 2] = (sin_theta_dt - sin_theta) / omega
    A[0, 3] = v * (cos_theta_dt - cos_theta) / omega
    A[1, 2] = (-cos_theta_dt + cos_theta) / omega
    A[1, 3] = v * (sin_theta_dt - sin_theta) / omega
    A[3, 4] = dt

    return torch.tensor(A, dtype=torch.float32)

class CTRVDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        # 原 xs: (N, T, 5), zs: (N, T, 2)
        self.xs = data['xs']
        self.zs = data['zs']

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        x_seq = torch.from_numpy(self.xs[idx]).float()  # (T,5)
        z_seq = torch.from_numpy(self.zs[idx]).float()  # (T,2)
        return x_seq, z_seq

def compute_structured_loss(
    x_pred,    # (B,5)
    x_true,    # (B,5)
    x_prev,    # (B,5)
    z_k,       # (B,2)         # (5,5)  torch.Tensor
    C,         # (2,5)  torch.Tensor
    Q,         # (5,5)  torch.Tensor
    R,         # (2,2)  torch.Tensor
    dt = 0.1,
    sigma_a=np.sqrt(0.01),
    sigma_alpha=np.sqrt(0.001),
    lambda_dyn=1.0,
    lambda_obs=1.0
):
    """
    1) ||x_pred - x_true||^2
    2) || (x_pred - A x_prev)(x_pred - A x_prev)^T - Q ||_F^2
    3) || (z_k - C x_pred)(z_k - C x_pred)^T - R ||_F^2
    最后各自做 batch 均值，再加权求和。
    """
    B, d = x_pred.shape
    m    = z_k.shape[1]

    # 1) 重构
    mse = nn.MSELoss(reduction='mean')
    loss_recon = mse(x_pred, x_true)

    # 2) 动力学外积差异
    v = x_prev[:, 2]
    theta = x_prev[:, 3]
    omega = x_prev[:, 4]
    eps = 1e-6
    omega = omega.clone().abs().clamp_min(eps) * torch.sign(omega)
    #a_long = np.random.normal(0, sigma_a)
    #a_ang = np.random.normal(0, sigma_alpha)
    x_next = torch.stack([
        x_prev[:,0] + (v / omega) * (torch.sin(theta + omega * dt) - torch.sin(theta)),
        x_prev[:,1] - (v / omega) * (torch.cos(theta + omega * dt) - torch.cos(theta)),
        v,
        theta + omega * dt,
        omega
    ],dim =1)

    e_dyn     = x_pred - x_next          # (B,5)
    dyn_outer = e_dyn.unsqueeze(2) * e_dyn.unsqueeze(1)  # (B,5,5)
    diff_dyn  = dyn_outer #- Q.unsqueeze(0)        # (B,5,5)
    loss_dyn = torch.sqrt(diff_dyn.pow(2).sum(dim=(1,2)) + 1e-12).mean()

    # 3) 观测外积差异
    e_obs     = z_k - (x_pred @ C.T)              # (B,2)
    obs_outer = e_obs.unsqueeze(2) * e_obs.unsqueeze(1)  # (B,2,2)
    diff_obs  = obs_outer #- R.unsqueeze(0)        # (B,2,2)
    loss_obs = torch.sqrt(diff_obs.pow(2).sum(dim=(1,2)) + 1e-12).mean()

    return   lambda_dyn * loss_dyn + lambda_obs * loss_obs


class PlanarFlow(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(z_dim))
        self.w = nn.Parameter(torch.randn(z_dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        # z: (B, z_dim)
        linear = z @ self.w + self.b             # (B,)
        activation = torch.tanh(linear)          # (B,)
        z_new = z + self.u * activation.unsqueeze(1)  # (B, z_dim)

        # 计算 log-det-Jacobian: log|1 + uᵀ ψ(z)|
        psi = (1 - activation**2).unsqueeze(1) * self.w  # (B, z_dim)
        det_jac = 1 + (psi * self.u).sum(dim=1)         # (B,)
        log_det_jac = torch.log(torch.abs(det_jac) + 1e-8)
        return z_new, log_det_jac

# --- 2. Conditional VAE definition ---
class StepVAE(nn.Module):
    def __init__(self, meas_dim=2, state_dim=5, h_dim=64, z_dim=16,n_flows=4):
        super().__init__()
        # 编码网络：上一状态 + 本测量 → 隐层
        self.enc = nn.Sequential(
            nn.Linear(state_dim+meas_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.fc_mu   = nn.Linear(h_dim, z_dim)
        self.fc_logv = nn.Linear(h_dim, z_dim)
        # 解码网络：z → 下一个状态
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, state_dim)
        )
        self.flows = nn.ModuleList([PlanarFlow(z_dim) for _ in range(n_flows)])

    def forward(self, prev_x, meas):
        """
        prev_x: (B, state_dim)
        meas:   (B, meas_dim)
        return: x_pred (B, state_dim), mu (B,z_dim), logv (B,z_dim)
        """
        inp = torch.cat([prev_x, meas], dim=1)    # (B, state_dim+meas_dim)
        h   = self.enc(inp)                       # (B, h_dim)
        mu  = self.fc_mu(h)                       # (B, z_dim)
        logv= self.fc_logv(h)                     # (B, z_dim)
        logv = torch.clamp(logv, min=-10, max=10)
        std = torch.exp(0.5*logv)
        #std = F.softplus(raw) + 1e-3
        #logv = 2.0 * torch.log(std.clamp_min(1e-12))  # ★ 用“有效”的 logv

        eps = torch.randn_like(std)
        z   = mu + eps * std                     # (B, z_dim)

        sum_log_det = 0.0
        for flow in self.flows:
            z, log_det = flow(z)
            sum_log_det = sum_log_det + log_det

        x_pred = self.dec(z)                      # (B, state_dim)
        #if torch.any(torch.isnan(x_pred)) or torch.any(torch.abs(x_pred) > 1e4):
            #print(f"Exploding prediction at step {k}: {x_pred}")
        return x_pred, mu, logv, sum_log_det

# --- 3. Training ---
def train_step_vae(
        train_loader,  # 传入 DataLoader
        epochs=30,
        dt=0.1,
        sigma_a = np.sqrt(0.01),
        sigma_alpha = np.sqrt(0.001),
        sigma_r=np.sqrt(0.5),
        sigma_phi = np.sqrt(0.05),
        lr=1e-4,
        print_every=1000):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StepVAE().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction='mean')


    C = torch.zeros(2, 5).to(device)
    C[0, 0] = 1.0  # y[0] = x
    C[1, 1] = 1.0  # y[1] = y

    R_polar = torch.diag(torch.tensor(
        [sigma_r**2, sigma_phi**2],
        dtype=torch.float32, device=device
    ))
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for x_batch, z_batch in train_loader:

            x_batch = x_batch.to(device)
            z_batch = z_batch.to(device)
            B, T, _ = x_batch.shape

            for b in range(B):
                xs = x_batch[b]  # (T,5)
                zs = z_batch[b]  # (T,2)

                r = zs[:, 0]
                phi = zs[:, 1]
                x_meas = r * torch.cos(phi)
                y_meas = r * torch.sin(phi)
                zs = torch.stack([x_meas, y_meas], dim=1)  # (T,2)


                prev_x = xs[0].unsqueeze(0)  # (1,5)
                total_loss_batch = 0;
                for k in range(1, T):
                    z_k = zs[k].unsqueeze(0)  # (1,2)
                    true_x = xs[k].unsqueeze(0)  # (1,5)
                    x_prior = ctrv_dynamics_torch(prev_x, dt, 0, 0)

                    x_pred, mu, logv, sum_log_det = model(prev_x, z_k)
                    # 重构 + KL
                     # 例如 从 0 → 1.0 线性/余弦
                    kl_dim = -0.5 * (1 + logv - mu.pow(2) - logv.exp())
                    kl_dim = torch.clamp(kl_dim, min=0.5)  # free-bits, 可先不开再逐步开
                    kl = kl_dim.mean() - torch.mean(sum_log_det)

                    #kl = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
                    #kl = kl - torch.mean(sum_log_det)

                    # 动态 Q / R 同前
                    theta = prev_x[0, 3].item()
                    Q_np = compute_process_noise_Q(theta, dt, sigma_a, sigma_alpha)
                    Q_t = torch.from_numpy(Q_np).to(device)
                    # polar->cart, J 和 R
                    r_k, phi_k = torch.norm(z_k), torch.atan2(z_k[0, 1], z_k[0, 0])
                    J = torch.tensor([
                        [phi_k.cos(), -r_k * phi_k.sin()],
                        [phi_k.sin(), r_k * phi_k.cos()]
                    ], device=device)
                    R_t = J @ R_polar @ J.t()

                    # 平滑项
                    loss_smooth = nn.functional.mse_loss(x_pred, prev_x)

                    # 结构化损失
                    loss_struct = compute_structured_loss(
                        x_pred, true_x, prev_x, z_k, C, Q_t, R_t
                    )

                    loss = loss_struct + 1e-4 * kl #1 * loss_smooth
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    prev_x = x_pred.detach()
                    total_loss += loss.item()
                    total_loss_batch += loss.item()
                    steps += 1
                
                print(f"Batch:{b}, Avg loss: {total_loss_batch / (T - 1) / (b+1)}")

        print(f">>> Epoch {ep} finished. Avg loss: {total_loss / (T - 1)/B:.6f}")

    torch.save(model.state_dict(), 'step_vae_whitenoise.pth')
    return model



if __name__ == "__main__":
    dataset = CTRVDataset('vae_dataset.npz')
    train_size = int(0.008 * len(dataset))
    test_size = int(0.002 * len(dataset))
    unused = len(dataset) - train_size - test_size
    train_ds, test_ds,_ = random_split(dataset, [train_size, test_size,unused])

    batch_size = 8
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = train_step_vae(
        train_loader=train_loader,
        epochs=10,
        dt=0.1,
        sigma_a=np.sqrt(0.01),
        sigma_alpha=np.sqrt(0.001),
        sigma_r=np.sqrt(0.5),
        sigma_phi=np.sqrt(0.05),
        lr=1e-4,
        print_every=500
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mse = nn.MSELoss(reduction='mean')
    model.eval()

    C = torch.zeros(2, 5, device=device)
    C[0, 0] = 1.0;
    C[1, 1] = 1.0

    all_sq_errors = []
    with torch.no_grad():
        for x_seq, z_seq in test_loader:
            x_seq = x_seq.to(device)  # (1, T, 5)
            z_seq = z_seq.to(device)  # (1, T, 2)
            T = x_seq.shape[1]

            # 用真值的第 0 步初始化
            prev_x = x_seq[:, 0, :]  # (1,5)

            preds = [prev_x[0].cpu().numpy()]
            for k in range(1, T):
                z_k = z_seq[:, k, :]  # (1,2)
                x_pred, _, _, _ = model(prev_x, z_k)
                preds.append(x_pred[0].cpu().numpy())
                prev_x = x_pred

            preds = np.stack(preds, axis=0)  # (T,5)
            true_vals = x_seq.cpu().numpy()  # (T,5)

            # 平方误差累积
            all_sq_errors.append((preds - true_vals) ** 2)

    all_sq_errors = np.concatenate(all_sq_errors, axis=0)  # (num_samples*T, 5)
    rmse = np.sqrt(np.mean(all_sq_errors))
    print(f"Test RMSE over all state dims: {rmse:.4f}")

    # --- 可视化轨迹 ---
    x_seq, z_seq = next(iter(test_loader))
    x_seq = x_seq.to(device);
    z_seq = z_seq.to(device)
    T = x_seq.shape[1]
    prev_x = x_seq[:, 0, :]
    preds = [ prev_x[0].cpu().numpy() ]
    for k in range(1, T):
        z_k = z_seq[:, k, :]
        x_pred, _, _, _ = model(prev_x, z_k)
        preds.append(x_pred[0].detach().cpu().numpy())
        prev_x = x_pred
    preds = np.stack(preds, axis=0)
    true_vals = x_seq.cpu().numpy()

    # 只取位置 (x, y) 两维作图
    plt.figure(figsize=(6, 6))
    plt.plot(true_vals[:, 0], true_vals[:, 1], label='True')
    plt.plot(preds[:, 0], preds[:, 1], '--', label='Pred')
    plt.xlabel('x');
    plt.ylabel('y');
    plt.axis('equal')
    plt.title('Example Test Trajectory')
    plt.legend()
    plt.show()