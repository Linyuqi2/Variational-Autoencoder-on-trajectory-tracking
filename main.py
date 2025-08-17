import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from case1_filters import *
from metrics import *
from vae import *


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
idx =990# np.random.randint(0, N)  # 随机挑一个 0 ~ N-1 的索引
xs= xs_all[idx]  # (T, 5)
zs = zs_all[idx]  # (T, 2)

print("chosen idx:", idx)


P0 = np.eye(5)
#Q  = np.diag([0.1, 0.1, 0.01, 0.01, 0.001])
R  = np.diag([0.5, 0.05])

xs_ekf = ekf_tracking(x0, P0, sigma_a,sigma_alpha, R, zs, dt)
xs_ukf = ukf_tracking(x0, P0, sigma_a,sigma_alpha, R, zs, dt)
xs_pf = pf_tracking(zs,x0,P0,500,dt, sigma_a, sigma_alpha, R)

name1 = 'step_vae_noQR_temperature.pth'
name2 = 'step_vae_noQR_temperature.pth'
#vae part
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StepVAE().to(device)
ckpt  = torch.load(name1, map_location=device)
model.load_state_dict(ckpt)
model.eval()

model2 = StepVAE().to(device)
ckpt2  = torch.load(name2, map_location=device)
model2.load_state_dict(ckpt2)
model2.eval()


r   = zs[:, 0]
phi = zs[:, 1]
x_meas = r * np.cos(phi)
y_meas = r * np.sin(phi)
zs_xy = np.stack([x_meas, y_meas], axis=1)  # (T,2)
meas_seq = torch.tensor(zs_xy, dtype=torch.float32).to(device)

preds    = np.zeros((N, 5), dtype=np.float32)
preds[0] = x0

preds2    = np.zeros((N, 5), dtype=np.float32)
preds2[0] = x0

prev_x = torch.tensor(x0, dtype=torch.float32).unsqueeze(0).to(device)  # (1,5)
prev_x2 = torch.tensor(x0, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    for k in range(N):
        z_k = meas_seq[k].unsqueeze(0)    # (1,2)
        x_pred, mu, logv, _ = model(prev_x, z_k)  # x_pred: (1,5)
        x_np = x_pred.squeeze(0).cpu().numpy()  # 转成 (5,)
        preds[k] = x_np
        prev_x = x_pred
    print("latent std mean:", (F.softplus(logv) + 1e-3).mean().item())
    print("logv:", logv)


with torch.no_grad():
    for k in range(N):
        z_k = meas_seq[k].unsqueeze(0)  # (1,2)
        x_pred2, mu2, logv2, _ = model2(prev_x2, z_k)  # x_pred: (1,5)
        x_np2 = x_pred2.squeeze(0).cpu().numpy()  # 转成 (5,)
        preds2[k] = x_np2
        prev_x2 = x_pred2

rmse = rmse_scalar(xs[:,:2], preds[:,:2])
print(name1+f"全轨迹位置 RMSE: {rmse:.4f}")
rmse2 = rmse_scalar(xs[:,:2], preds2[:,:2])
print(name2+f"全轨迹位置 RMSE: {rmse2:.4f}")

rmse_total_ekf = rmse_scalar(xs[:,:2], xs_ekf[:,:2])
rmse_total_ukf = rmse_scalar(xs[:,:2], xs_ukf[:,:2])
rmse_total_pf = rmse_scalar(xs[:,:2], xs_pf[:,:2])
print("scalar RMSE of position: EKF: ",rmse_total_ekf,"UKF: ",rmse_total_ukf,"PF: ",rmse_total_pf)


# 6. 可视化
plt.figure(figsize=(6,6))
plt.plot(xs[:,0], xs[:,1], '-k', label='True trajectory')
xy_meas = np.vstack([zs[:,0]*np.cos(zs[:,1]),
                     zs[:,0]*np.sin(zs[:,1])]).T
plt.scatter(xy_meas[:,0], xy_meas[:,1],s=1, c='r', label='Measurements')

#plt.plot(xs_ekf[:,0], xs_ekf[:,1], '--b', label='EKF')
plt.scatter(preds2[:,0], preds2[:,1], 1,'g', label=name2)
#plt.plot(xs_pf[:,0], xs_pf[:,1], '--y', label='PF')
plt.scatter(preds[:,0], preds[:,1], 1,'b', label=name1)
#print(preds[:,0])
plt.legend(); plt.axis('equal')
plt.xlabel('x'); plt.ylabel('y')
plt.title('CTRV Simulation with Process & Measurement Noise')
plt.grid(True)
plt.show()
