import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat

import logging
import json as _json
import os
import time
from datetime import datetime

# ── Run bookkeeping ────────────────────────────────────────────────────────
RUN_ID   = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR  = os.path.join('runs', 'LiNO_log')
CKPT_DIR = os.path.join('runs', 'LiNO_log')
os.makedirs(CKPT_DIR, exist_ok=True)

# ── log ─────────────────────────────────────────────────────────────────
log_path = os.path.join(OUT_DIR, 'train.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | INFO | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_path, mode='a'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('LiNO')


best_test_rel_l2 = float('inf')
best_ckpt_path   = os.path.join(CKPT_DIR, 'trained_model.pth')
t0 = time.time()

# ── Config ─────────────────────────────────────────────────────────────────
seed   = 42
ntrain = 800
ntest  = 200
levels = 4
epochs = 500
lr     = 1e-3
width  = 32
batch_size = 128
data_path  = '/home/phimanshu/SSD/Shared/SP_HP_shared/Data/poisson/poisson_256_256.npz'
S = 2   

config = dict(
    run_id=RUN_ID, seed=seed, ntrain=ntrain, ntest=ntest,
    levels=levels, epochs=epochs, lr=lr, width=width,
    batch_size=batch_size, data_path=data_path
)
config_path = os.path.join(OUT_DIR, 'config.json')
with open(config_path, 'w') as fp:
    _json.dump(config, fp, indent=2)

log.info(f'Run ID : {RUN_ID}')
log.info(f'Output : {OUT_DIR}')
log.info(f'Seed   : {seed}')
log.info(f'Config saved \u2192 {config_path}')

# ── Reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
log.info(f'Device : {device}')

# ── Data ───────────────────────────────────────────────────────────────────
log.info(f'[1/4] Loading data from {data_path}')
data = np.load(data_path)
sub  = S

a = torch.tensor(data["F"][:, ::sub, ::sub],      dtype=torch.float32).unsqueeze(1)
u = torch.tensor(data["U"][:, ::sub, ::sub], dtype=torch.float32).unsqueeze(1)
log.info(f"   a shape : {tuple(a.shape)}   u shape : {tuple(u.shape)}")

a_mean, a_std = a.mean(), a.std()
u_mean, u_std = u.mean(), u.std()
a = (a - a_mean) / a_std
u = (u - u_mean) / u_std


log.info(f'   a shape : {tuple(a.shape)}   u shape : {tuple(u.shape)}')
log.info(f'   Train : {ntrain} | Test : {ntest}')

a_train, a_test = a[:ntrain], a[ntrain:ntrain+ntest]
u_train, u_test = u[:ntrain], u[ntrain:ntrain+ntest]

train_loader = DataLoader(TensorDataset(a_train, u_train),
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(a_test,  u_test),
                          batch_size=32)


def add_coords(x):
    B, C, H, W = x.shape
    # print(x.shape)
    gx = torch.linspace(0,1,H, device=x.device)
    gy = torch.linspace(0,1,W, device=x.device)
    gx, gy = torch.meshgrid(gx, gy, indexing='ij')

    grid = torch.stack([gx, gy], dim=0)
    grid = grid.unsqueeze(0).repeat(B,1,1,1)

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
    def __init__(self, in_ch, out_ch, hidden=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
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
    def __init__(self, channels, levels=4):
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

        return x, coeffs  

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
        return self.net(x) + self.W(x)
  



class LiNO(nn.Module):
    def __init__(self, in_channels=1, width=64, levels=4):
        super().__init__()

        # self.W = nn.Conv2d(width, width, 1)

        self.lift = nn.Conv2d(in_channels+2, width, 1)

        self.lifting = MultiLevelLifting(width, levels)

        self.operator_coarse = CoefficientOperator(width)

        self.op_h = CoefficientOperator(width)  # horizontal (d01)
        self.op_v = CoefficientOperator(width)  # vertical (d10)
        self.op_d = CoefficientOperator(width)  # diagonal (d11)

        self.proj = nn.Conv2d(width, 1, 1)



    def forward(self, x):

        x_in = x

        x = add_coords(x)  # (B, C+2, H, W)
        x = self.lift(x)
        

        # Lifting decomposition
        s, coeffs = self.lifting(x)

        # Apply operator
        s = self.operator_coarse(s)
        new_coeffs = []
        for d_tuple in coeffs:
            d01, d10, d11 = d_tuple

            d01 = self.op_h(d01)
            d10 = self.op_v(d10)
            d11 = self.op_d(d11)


            new_coeffs.append((d01, d10, d11))

        coeffs = new_coeffs

        # Reconstruction
        x_recon = self.lifting.inverse(s, coeffs) 

        x = self.proj(x_recon)
        
        return x


log.info('[2/4] Building LiNO \u2026')

model = LiNO(in_channels=1, width=width, levels=levels).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f'   Trainable parameters : {n_params:,}')

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-5
)

def loss_fn(pred, target):
    return torch.norm(pred - target) / (torch.norm(target) + 1e-8)


def model_eval(epoch, loss, t0=t0, inference=False):
    global best_test_rel_l2
    model.eval()
    with torch.no_grad():
        rel_errors = []
        for idx in range(len(a_test)):
            a_s = a_test[idx:idx+1].to(device)
            u_s = u_test[idx:idx+1].to(device)
            p_s = model(a_s)
            p_s = p_s * u_std + u_mean
            u_s = u_s * u_std + u_mean
            err = loss_fn(p_s, u_s)
            rel_errors.append(err.item())

    rel_errors_t = torch.tensor(rel_errors)
    test_mean = rel_errors_t.mean().item()
    test_std  = rel_errors_t.std().item()
    elapsed = (time.time() - t0) / 60
    log.info(
        f'  Epoch {epoch:4d}/{epochs} | '
        f'Loss {loss:.4e} | '
        f'Test rel-L2 {test_mean:.4e} \u00b1 {test_std:.4e} | '
        f'Elapsed {elapsed:.2f} min'
    )

    # ── save best model ───────────────────────────────────────────────
    if test_mean < best_test_rel_l2 and inference == False:
        best_test_rel_l2 = test_mean
        torch.save({
            'epoch':          epoch,
            'model_state':    model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'test_rel_l2':    test_mean,
            'test_std':       test_std,
            'config':         config,
            'loss':           loss
        }, best_ckpt_path)

    torch.cuda.empty_cache()

def train():
    for epoch in tqdm(range(epochs)):
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total_loss = 0.0

        for a_batch, u_batch in train_loader:
            a_batch = a_batch.to(device)
            u_batch = u_batch.to(device)

            pred = model(a_batch)
            loss = loss_fn(pred, u_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch < epochs:
            scheduler.step()

        train_rel_l2 = total_loss / len(train_loader)

        if epoch % 50 == 0 or epoch == epochs - 1:
            model_eval(epoch, train_rel_l2)
        

    log.info(f'[4/4] Training complete. Best test rel-L2 = {best_test_rel_l2:.4e}')
    log.info(f'      Best checkpoint  \u2192 {best_ckpt_path}')
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  
    log.info(f'      Peak GPU memory usage : {peak_mem_mb:.2f} GB')


def visualization():
    TITLE_FONTSIZE = 18
    CBAR_FONTSIZE = 14

    model.eval()
    sample_idx = np.random.choice(len(a_test), size=3, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    with torch.no_grad():
        for row, idx in enumerate(sample_idx):
            a_s = a_test[idx:idx+1].to(device)
            u_s = u_test[idx:idx+1].to(device)
            p_s = model(a_s)

            p_s = (p_s * u_std + u_mean).cpu().squeeze().numpy()
            u_s = (u_s * u_std + u_mean).cpu().squeeze().numpy()
            err = np.abs(p_s - u_s)

            im0 = axes[row, 0].imshow(u_s, cmap='viridis')
            axes[row, 0].set_title(f'Ground Truth (idx={idx})', fontsize=TITLE_FONTSIZE)
            cbar0 = plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)
            cbar0.ax.tick_params(labelsize=CBAR_FONTSIZE)

            im1 = axes[row, 1].imshow(p_s, cmap='viridis')
            axes[row, 1].set_title(f'Prediction (idx={idx})', fontsize=TITLE_FONTSIZE)
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