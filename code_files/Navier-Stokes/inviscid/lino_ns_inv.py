import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.io import loadmat
import h5py
import os


CKPT_DIR = os.path.join('model')
os.makedirs(CKPT_DIR, exist_ok=True)
best_ckpt_path   = os.path.join(CKPT_DIR, 'trained_model.pth')
t0 = time.time()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


T_in    = 8
T_out   = 13
step    = 1
sample  = 1000
levels  = 3
ntrain  = 800
ntest   = 200
epochs  = 500
sub = 1
DENSITY_IDX = 2   # index among [Vx, Vy, density, pressure]
N_FIELDS    = 4

data_path = "/home/phimanshu/SSD/Shared/SP_HP_shared/Data/CFD/2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Train.hdf5"
data = h5py.File(data_path, "r")
print("Keys:", list(data.keys()))
print(data['Vx'].shape)  



Vx  = torch.tensor(data['Vx'      ][:sample,:,::sub,::sub], dtype=torch.float32)
Vy  = torch.tensor(data['Vy'      ][:sample,:,::sub,::sub], dtype=torch.float32)
rho = torch.tensor(data['density' ][:sample,:,::sub,::sub], dtype=torch.float32)
p   = torch.tensor(data['pressure'][:sample,:,::sub,::sub], dtype=torch.float32)



# Stack -> (N, 4, T, X, Y)
u_all  = torch.stack([Vx, Vy, rho, p], dim=1)
N, C, T_total, X, Y = u_all.shape

# Per-channel normalisation — shape (1, 4, 1, 1, 1)
u_mean = u_all.mean(dim=(0, 2, 3, 4), keepdim=True)
u_std  = u_all.std( dim=(0, 2, 3, 4), keepdim=True)
u_norm = (u_all - u_mean) / (u_std + 1e-8)   # (N, 4, T_total, X, Y)

u_train = u_norm[:ntrain]
u_test  = u_norm[ntrain:ntrain+ntest]

# Each item is the full time series — slicing done inside training loop
train_loader = DataLoader(TensorDataset(u_train), batch_size=20, shuffle=True)
test_loader  = DataLoader(TensorDataset(u_test),  batch_size=16)

print("u_train shape:", u_train.shape)  # (ntrain, 4, T, X, Y)
print("u_test shape:", u_test.shape)    # (ntest, 4, T, X, Y)

_coord_cache = {}
def add_coords(x):
    B, C, H, W = x.shape
    key = (H, W, x.device)
    if key not in _coord_cache:
        gx = torch.linspace(0,1,H, device=x.device)
        gy = torch.linspace(0,1,W, device=x.device)
        gx, gy = torch.meshgrid(gx, gy, indexing='ij')
        grid = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1,2,H,W)
        _coord_cache[key] = grid
    grid = _coord_cache[key].expand(B, -1, -1, -1)
    return torch.cat([x, grid], dim=1)


class Blur(nn.Module):
    def __init__(self, channels):
        super().__init__()

        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]]) / 16.0

        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1, groups=self.channels)
    

class LocalMLP(nn.Module):
    def __init__(self, in_ch, out_ch, hidden=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, padding_mode='circular'),
            nn.GELU(),

            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'),
            nn.GELU(),

            nn.Conv2d(hidden, out_ch, 1)
        )

        self.proj = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.proj(x) + self.net(x)
    


class LiftingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        in_ch = channels + 2

        self.P1 = LocalMLP(in_ch, channels)
        self.P2 = LocalMLP(in_ch, channels)
        self.P3 = LocalMLP(in_ch, channels)

        self.U = LocalMLP(3*in_ch, channels)
        self.blur = Blur(channels)

    def forward(self, x):

        x = self.blur(x)

        x00 = x[:, :, ::2, ::2]
        x01 = x[:, :, ::2, 1::2]
        x10 = x[:, :, 1::2, ::2]
        x11 = x[:, :, 1::2, 1::2]


        x00_c = add_coords(x00)

        d01 = x01 - self.P1(x00_c)
        d10 = x10 - self.P2(x00_c)
        d11 = x11 - self.P3(x00_c)

        d01_c = add_coords(d01)
        d10_c = add_coords(d10)
        d11_c = add_coords(d11)

        d_cat = torch.cat([d01_c, d10_c, d11_c], dim=1)

        s = x00 + self.U(d_cat)

        return s, (d01, d10, d11)

    def inverse(self, s, d_tuple):
        d01, d10, d11 = d_tuple

        d01_c = add_coords(d01)
        d10_c = add_coords(d10)
        d11_c = add_coords(d11)

        d_cat = torch.cat([d01_c, d10_c, d11_c], dim=1)

        x00 = s - self.U(d_cat)

        x00_c = add_coords(x00)

        x01 = d01 + self.P1(x00_c)
        x10 = d10 + self.P2(x00_c)
        x11 = d11 + self.P3(x00_c)

        B, C, H, W = x00.shape
        out = torch.zeros(B, C, H*2, W*2, device=s.device)

        out[:, :, ::2, ::2] = x00
        out[:, :, ::2, 1::2] = x01
        out[:, :, 1::2, ::2] = x10
        out[:, :, 1::2, 1::2] = x11

        return out


class MultiLevelLifting(nn.Module):
    def __init__(self, channels, levels):
        super().__init__()
        self.levels = levels
        self.blocks = nn.ModuleList([
            LiftingBlock(channels) for _ in range(levels)
        ])

    def forward(self, x):
        coeffs = []

        for i in range(self.levels):
            s, d = self.blocks[i](x)
            coeffs.append(d)
            x = s

        return x, coeffs  # coarse + details

    def inverse(self, s, coeffs):
        for i in reversed(range(self.levels)):
            s = self.blocks[i].inverse(s, coeffs[i])
        return s



class CoefficientOperator(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.W = nn.Conv2d(channels, channels, 1)  

    def forward(self, x):
        return self.net(x) + self.W(x)   # residualclass CoefficientOperator(nn.Module):
  



class LiNO(nn.Module):
    def __init__(self, in_channels, out_channels, width, levels):
        super().__init__()
        self.lift     = nn.Conv2d(in_channels + 2, width, 1)
        self.lifting  = MultiLevelLifting(width, levels)

        self.operator_coarse = CoefficientOperator(width)
        self.op_h = CoefficientOperator(width)   # horizontal details
        self.op_v = CoefficientOperator(width)   # vertical details
        self.op_d = CoefficientOperator(width)   # diagonal details

        self.proj = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        # x: (B, N_FIELDS*T_in, X, Y)
        x = add_coords(x)       # (B, N_FIELDS*T_in+2, X, Y)
        x = self.lift(x)        # (B, width, X, Y)

        s, coeffs = self.lifting(x)
        s = self.operator_coarse(s)

        new_coeffs = []
        for (d01, d10, d11) in coeffs:
            new_coeffs.append((self.op_h(d01), self.op_v(d10), self.op_d(d11)))

        x_recon = self.lifting.inverse(s, new_coeffs)
        return self.proj(x_recon)   # (B, N_FIELDS, X, Y)



model = LiNO(in_channels=N_FIELDS * T_in, out_channels=N_FIELDS, width=64, levels=levels).to(device)
print(f"LiNO parameter count: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-5
)



def loss_fn(pred, target):
    rel_l2 = torch.norm(pred - target) / (torch.norm(target) + 1e-8)

    return rel_l2


def model_eval(epoch, loss, t0=t0, inference=False):
    model.eval()

    n_samples = u_test.shape[0]
    total_rel_l2_rho = 0.0

    rho_mean = u_mean[:, DENSITY_IDX, :, :, :].to(device)   # (1, 1, 1, 1) → broadcasts over (X, Y)
    rho_std  = u_std[ :, DENSITY_IDX, :, :, :].to(device)
    std = []

    with torch.no_grad():
        for i in range(n_samples):
            u_s  = u_test[i:i+1].to(device)               # (1, 4, T_total, X, Y)

            # xx_r = u_s[:, :, :T_in].reshape(1, N_FIELDS * T_in, X, Y)
            xx_r = (u_s[:, :, :T_in].permute(0, 2, 1, 3, 4).reshape(u_s.shape[0], T_in * N_FIELDS, X, Y))
            yy_t = u_s[:, :, T_in:T_in + T_out]           # (1, 4, T_out, X, Y)

            preds = []
            for t in range(0, T_out, step):
                im = model(xx_r)                           # (1, 4, X, Y)
                preds.append(im.unsqueeze(2))              # (1, 4, 1, X, Y)
                xx_r = torch.cat([xx_r[:, N_FIELDS:], im], dim=1)

            pred_seq = torch.cat(preds, dim=2)             # (1, 4, T_out, X, Y)

            # Extract density channel and denormalise
            pred_rho = pred_seq[:, DENSITY_IDX] * rho_std + rho_mean   # (1, T_out, X, Y)
            true_rho = yy_t[   :, DENSITY_IDX] * rho_std + rho_mean

            rel_l2 = loss_fn(pred_rho, true_rho)
            total_rel_l2_rho += rel_l2.item()
            std.append(rel_l2.item())

    test_mean = total_rel_l2_rho / n_samples
    test_std = torch.tensor(std).std().item()
    t1 = time.time()
    elapsed = (t1 - t0) / 60

    print(f"Epoch {epoch:4d}  Loss: {loss:.6f}  "
            f"Density RelL2: {test_mean:.6f}, Std: {test_std:.6f},    Time: {elapsed:.2f} min")

    # torch.save(model.state_dict(), os.path.join(CKPT_DIR, "lino_cfd_2d_best.pth"))    
    if inference == False:
        torch.save({
                    'epoch':          epoch,
                    'model_state':    model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'test_rel_l2':    test_mean,
                    'test_std':       test_std,
                    'loss':           loss
                }, best_ckpt_path)
    
    torch.cuda.empty_cache()

def train():
    prev_loss = 0
    for epoch in range(epochs):
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total_loss = 0

        for u_batch, in train_loader:

            u_batch = u_batch.to(device)   # (B, T, H, W)

            xx = (u_batch[:, :, :T_in].permute(0, 2, 1, 3, 4).reshape(u_batch.shape[0], T_in * N_FIELDS, X, Y))
            yy = u_batch[:, :, T_in:T_in + T_out]

            loss = 0
            for t in range(0, T_out, step):
                y  = yy[:, :, t] 
                im = model(xx)
                loss += loss_fn(im, y)
                # autoregressive update
                xx = torch.cat([xx[:, N_FIELDS:], im], dim=1)

            loss = loss / T_out
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_rel_l2 = total_loss/len(train_loader)

        scheduler.step()
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            model_eval(epoch, train_rel_l2)

            

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  
    print(f'Peak GPU memory usage : {peak_mem_mb:.2f} GB')


def visualization():
    TITLE_FONTSIZE = 18
    CBAR_FONTSIZE = 14

    model.eval()

    n_samples = u_test.shape[0]
    sample_idx = np.random.choice(n_samples, size=3, replace=False)

    rho_mean = u_mean[:, DENSITY_IDX, :, :, :].to(device)
    rho_std  = u_std[ :, DENSITY_IDX, :, :, :].to(device)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    with torch.no_grad():
        for row, idx in enumerate(sample_idx):
            u_s = u_test[idx:idx+1].to(device)

            xx_r = (u_s[:, :, :T_in].permute(0, 2, 1, 3, 4)
                     .reshape(u_s.shape[0], T_in * N_FIELDS, X, Y))
            yy_t = u_s[:, :, T_in:T_in + T_out]

            pred_last = None
            for t in range(0, T_out, step):
                im = model(xx_r)
                pred_last = im
                xx_r = torch.cat([xx_r[:, N_FIELDS:], im], dim=1)

            pred_rho = (pred_last[:, DENSITY_IDX] * rho_std + rho_mean).cpu().squeeze().numpy()
            true_rho = (yy_t[:, DENSITY_IDX, -1]  * rho_std + rho_mean).cpu().squeeze().numpy()
            err = np.abs(pred_rho - true_rho)

            im0 = axes[row, 0].imshow(true_rho, cmap='viridis')
            axes[row, 0].set_title(f'Ground Truth (idx={idx}, t={T_in+T_out-1})', fontsize=TITLE_FONTSIZE)
            cbar0 = plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)
            cbar0.ax.tick_params(labelsize=CBAR_FONTSIZE)

            im1 = axes[row, 1].imshow(pred_rho, cmap='viridis')
            axes[row, 1].set_title(f'Prediction (idx={idx}, t={T_in+T_out-1})', fontsize=TITLE_FONTSIZE)
            cbar1 = plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)
            cbar1.ax.tick_params(labelsize=CBAR_FONTSIZE)

            im2 = axes[row, 2].imshow(err, cmap='inferno')
            axes[row, 2].set_title(f'Absolute Error (idx={idx})', fontsize=TITLE_FONTSIZE)
            cbar2 = plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)
            cbar2.ax.tick_params(labelsize=CBAR_FONTSIZE)
            cbar2.update_ticks()

            for col in range(3):
                axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()