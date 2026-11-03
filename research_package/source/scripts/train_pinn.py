"""G1 Inverse Dynamics PINN — Adam vs Ensemble vs ParVI comparison."""
import torch, torch.nn as nn, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, os, time, json
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr

device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = "/root/unitree_rl_mjlab/results"
os.makedirs(out_dir, exist_ok=True)
mse_fn = nn.MSELoss()

# ═══════════════════════════════ 1. Data ═══════════════════════════════
print("Loading data...")
data = np.load("/root/unitree_rl_mjlab/data/g1_trajectory.npz")
q_raw = torch.tensor(data["q"], dtype=torch.float32)
qd_raw = torch.tensor(data["qd"], dtype=torch.float32)
qdd_raw = torch.tensor(data["qdd"], dtype=torch.float32)
tau_raw = torch.tensor(data["tau"], dtype=torch.float32)

def normed(x):
    xf = x.reshape(-1, x.shape[-1])
    m, s = xf.mean(0), xf.std(0).clamp(1e-6)
    return xf, m.to(device), s.to(device)

q, q_m, q_s = normed(q_raw); qd, qd_m, qd_s = normed(qd_raw)
qdd, qdd_m, qdd_s = normed(qdd_raw); tau, tau_m, tau_s = normed(tau_raw)
X = torch.cat([q, qd, qdd], -1).to(device); Y = tau.to(device)

n = X.shape[0]; n_tr = int(n*0.7); n_vl = int(n*0.15)
perm = torch.randperm(n)
X_tr, Y_tr = X[perm[:n_tr]], Y[perm[:n_tr]]
X_vl, Y_vl = X[perm[n_tr:n_tr+n_vl]], Y[perm[n_tr:n_tr+n_vl]]
X_te, Y_te = X[perm[n_tr+n_vl:]], Y[perm[n_tr+n_vl:]]
tr_ldr = DataLoader(TensorDataset(X_tr, Y_tr), 512, shuffle=True)
print(f"Train:{X_tr.shape[0]:,} Val:{X_vl.shape[0]:,} Test:{X_te.shape[0]:,}")

# ═══════════════════════════════ 2. Model ═══════════════════════════════
class PINN(nn.Module):
    def __init__(self, in_d=87, out_d=29, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, h), nn.SiLU(),
            nn.Linear(h, h),   nn.SiLU(),
            nn.Linear(h, h),   nn.SiLU(),
            nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)

def evaluate(model, X_, Y_, bs=2048):
    model.eval()
    loss, n_ = 0, X_.shape[0]
    with torch.no_grad():
        for i in range(0, n_, bs):
            xb, yb = X_[i:i+bs], Y_[i:i+bs]; loss += mse_fn(model(xb), yb).item()*xb.shape[0]
    return loss/n_

def train_one(seed, epochs=250, verbose=True):
    torch.manual_seed(seed)
    model = PINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_ldr:
            opt.zero_grad(); loss = mse_fn(model(xb), yb); loss.backward(); opt.step()
        sc.step()
    vl = evaluate(model, X_vl, Y_vl); te = evaluate(model, X_te, Y_te)
    if verbose: print(f"  seed={seed:2d} val={vl:.4f} test={te:.4f}")
    return model, vl, te

os.makedirs("/root/unitree_rl_mjlab/models", exist_ok=True)

# ═══════════════════════════════ 3. Single Adam ═══════════════════════════════
print("\n=== 1. Single Adam Baseline ===")
t0 = time.time()
adam_model, adam_vl, adam_te = train_one(42, 250)
adam_t = time.time()-t0
torch.save(adam_model.state_dict(), "/root/unitree_rl_mjlab/models/pinn_adam.pt")
print(f"  Test MSE: {adam_te:.6f} ({adam_t:.0f}s)")

# ═══════════════════════════════ 4. Deep Ensemble (5 models) ═══════════════════════
print("\n=== 2. Deep Ensemble (5 models) ===")
t0 = time.time()
ensemble = []
for s in [1, 2, 3, 4, 5]:
    m, vl, te = train_one(s, 250)
    ensemble.append(m)
    torch.save(m.state_dict(), f"/root/unitree_rl_mjlab/models/pinn_ens_{s}.pt")
ens_t = time.time()-t0

# Ensemble prediction
with torch.no_grad():
    ens_preds = torch.stack([m(X_te) for m in ensemble])  # [5, N, 29]
    ens_mean = ens_preds.mean(0)
    ens_std = ens_preds.std(0)
    ens_mse = mse_fn(ens_mean, Y_te).item()
    ens_unc = ens_std.mean(-1).cpu().numpy()  # per-sample uncertainty
    ens_err = (ens_mean - Y_te).pow(2).mean(-1).cpu().numpy()  # per-sample error
    ens_corr, ens_p = spearmanr(ens_err, ens_unc)

# Individual MSEs
ens_indiv = [evaluate(m, X_te, Y_te) for m in ensemble]
print(f"  Individual MSEs: {[f'{x:.4f}' for x in ens_indiv]}")
print(f"  Ensemble MSE: {ens_mse:.6f}")
print(f"  Avg individual: {np.mean(ens_indiv):.4f} → Ensemble: {ens_mse:.4f} (Δ={np.mean(ens_indiv)-ens_mse:.4f})")
print(f"  Uncertainty ρ: {ens_corr:.3f} (p={ens_p:.4f})")

# ═══════════════════════════════ 5. ParVI (bias-only) ═══════════════════════
print("\n=== 3. ParVI (bias-only, 29D, 50 particles) ===")
torch.manual_seed(99)
pv = PINN().to(device)
# Pretrain
opt_pv = torch.optim.Adam(pv.parameters(), lr=1e-3)
sc_pv = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pv, 60)
for ep in range(60):
    pv.train()
    for xb, yb in tr_ldr:
        opt_pv.zero_grad(); loss = mse_fn(pv(xb), yb); loss.backward(); opt_pv.step()
    sc_pv.step()
print(f"  Pretrain val MSE: {evaluate(pv, X_vl, Y_vl):.4f}")

# Freeze weights, ParVI on bias only
for p in pv.parameters(): p.requires_grad = False
last = list(pv.net.modules())[-1]
last.bias.requires_grad = True
W_fixed = last.weight.data.clone()
b_shape = last.bias.shape; D = b_shape[0]  # 29

M = 60; lr_p = 0.02; alpha = 0.5
particles = last.bias.data.flatten().unsqueeze(0) + torch.randn(M, D, device=device)*0.03
particles_prev = particles.clone()

t_pv = time.time()
for it in range(250):
    xb, yb = next(iter(tr_ldr))
    grads = torch.zeros(M, D, device=device)
    for i in range(M):
        last.bias.data = particles[i].reshape(b_shape); last.weight.data = W_fixed
        last.bias.grad = None; last.bias.requires_grad_(True)
        loss = mse_fn(pv(xb), yb); loss.backward()
        grads[i] = last.bias.grad.data.flatten().clone()
        last.bias.requires_grad_(False)

    with torch.no_grad():
        diff = particles.unsqueeze(0) - particles.unsqueeze(1)
        dist_sq = (diff**2).sum(-1)
        mask = torch.triu(torch.ones(M, M, device=device), 1) > 0
        med = dist_sq[mask].median().clamp(min=1e-4)
        k = torch.exp(-dist_sq/(2*med))
        repulse = (k.unsqueeze(-1)*diff/med).sum(1)
        rn = repulse.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        repulse = repulse / rn.clamp(max=3.0)

    new_p = particles - lr_p*grads + lr_p*alpha*repulse
    if new_p.isnan().any(): new_p = particles_prev; lr_p *= 0.5
    particles_prev = particles.clone(); particles = new_p.clone()

    if (it+1)%50==0:
        # Quick eval
        ps = []
        for i in range(M):
            last.bias.data = particles[i].reshape(b_shape); last.weight.data = W_fixed
            with torch.no_grad(): ps.append(pv(X_vl[:200]))
        ps = torch.stack(ps)
        print(f"  iter {it+1:3d} vl={mse_fn(ps.mean(0), Y_vl[:200]).item():.4f} med_dist={med.item():.4f}")

# Best particle
best_i, best_v = 0, float("inf")
for i in range(M):
    last.bias.data = particles[i].reshape(b_shape); last.weight.data = W_fixed
    vl = evaluate(pv, X_vl, Y_vl)
    if vl < best_v: best_v, best_i = vl, i
last.bias.data = particles[best_i].reshape(b_shape); last.weight.data = W_fixed

parvi_t = time.time()-t_pv
parvi_te = evaluate(pv, X_te, Y_te)

# ParVI ensemble prediction
with torch.no_grad():
    pv_preds = []
    for i in range(M):
        last.bias.data = particles[i].reshape(b_shape); last.weight.data = W_fixed
        pv_preds.append(pv(X_te))
    pv_preds = torch.stack(pv_preds)
    pv_mean = pv_preds.mean(0); pv_std = pv_preds.std(0)
    pv_ens_mse = mse_fn(pv_mean, Y_te).item()
    pv_unc = pv_std.mean(-1).cpu().numpy()
    pv_err = (pv_mean - Y_te).pow(2).mean(-1).cpu().numpy()
    pv_corr, pv_p = spearmanr(pv_err, pv_unc)

torch.save(pv.state_dict(), "/root/unitree_rl_mjlab/models/pinn_parvi.pt")
print(f"  ParVI best test MSE: {parvi_te:.4f}")
print(f"  ParVI ensemble MSE: {pv_ens_mse:.4f}")
print(f"  ParVI uncertainty ρ: {pv_corr:.3f}")

# ═══════════════════════════════ 6. Results ═══════════════════════════
print("\n"+"="*65)
print("FINAL RESULTS: G1 Inverse Dynamics PINN Training Methods")
print("="*65)
print(f"{'Method':<22} {'Test MSE':<12} {'Uncertainty':<15} {'Time':<8}")
print("-"*57)
print(f"{'1. Single Adam':<22} {adam_te:<12.4f} {'N/A':<15} {adam_t:<8.0f}s")
print(f"{'2. Deep Ensemble (5)':<22} {ens_mse:<12.4f} {ens_corr:<15.3f} {ens_t:<8.0f}s")
print(f"{'3. ParVI (bias, 60p)':<22} {pv_ens_mse:<12.4f} {pv_corr:<15.3f} {parvi_t:<8.0f}s")

print(f"\nKey findings:")
print(f"  Ensemble Δ: {np.mean(ens_indiv)-ens_mse:.4f} (variance reduction)")
print(f"  Ensemble uncertainty calibration: ρ={ens_corr:.3f}")
print(f"  ParVI uncertainty calibration: ρ={pv_corr:.3f}")

res = {
    "adam_test_mse": float(adam_te),
    "ensemble_test_mse": float(ens_mse),
    "ensemble_avg_individual_mse": float(np.mean(ens_indiv)),
    "ensemble_uncertainty_rho": float(ens_corr),
    "parvi_test_mse": float(parvi_te),
    "parvi_ensemble_mse": float(pv_ens_mse),
    "parvi_uncertainty_rho": float(pv_corr),
    "parvi_n_particles": M, "parvi_dim": D,
}
json.dump(res, open(f"{out_dir}/results.json","w"), indent=2)

# ═══════════════════════════════ 7. Plots ═══════════════════════════════
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

# 7a. Bar chart: MSE comparison
methods = ["Single\nAdam", "Deep\nEnsemble", "ParVI\n(bias)"]
mses = [adam_te, ens_mse, pv_ens_mse]
colors = ["#3498db", "#2ecc71", "#e74c3c"]
bars = axs[0].bar(methods, mses, color=colors, width=0.5, edgecolor="white")
for b, v in zip(bars, mses): axs[0].text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.4f}", ha="center", fontweight="bold", fontsize=11)
axs[0].set(ylabel="Test MSE", title="Prediction Accuracy")
axs[0].grid(alpha=0.3, axis="y")

# 7b. Uncertainty calibration — Ensemble
bins = 8
unc_b = np.percentile(ens_unc, np.linspace(0, 100, bins+1))
be, bu = [], []
for j in range(bins):
    mask = (ens_unc >= unc_b[j]) & (ens_unc < unc_b[j+1])
    if mask.sum() > 5: be.append(ens_err[mask].mean()); bu.append(ens_unc[mask].mean())
if len(be) > 2:
    axs[1].plot(bu, be, "o-", color="#2ecc71", ms=10, label="Ensemble")
    z = np.polyfit(bu, be, 1)
    axs[1].plot(bu, np.polyval(z, bu), "--", color="gray", alpha=0.5, lw=2)

bins2 = 6
unc_b2 = np.percentile(pv_unc, np.linspace(0, 100, bins2+1))
be2, bu2 = [], []
for j in range(bins2):
    mask = (pv_unc >= unc_b2[j]) & (pv_unc < unc_b2[j+1])
    if mask.sum() > 5: be2.append(pv_err[mask].mean()); bu2.append(pv_unc[mask].mean())
if len(be2) > 2:
    axs[1].plot(bu2, be2, "s-", color="#e74c3c", ms=10, label="ParVI")
axs[1].set(xlabel="Predicted Uncertainty (σ)", ylabel="Actual MSE",
           title=f"Uncertainty Calibration")
axs[1].legend(); axs[1].grid(alpha=0.3)

# 7c. Ensemble scatter
samp = min(2000, len(ens_err))
axs[2].scatter(ens_unc[:samp], ens_err[:samp], alpha=0.35, s=6, color="#2ecc71",
               label=f"Ensemble ρ={ens_corr:.3f}")
axs[2].scatter(pv_unc[:samp], pv_err[:samp], alpha=0.25, s=6, color="#e74c3c",
               label=f"ParVI ρ={pv_corr:.3f}")
axs[2].set(xlabel="Uncertainty (σ)", ylabel="Per-sample MSE",
           title="Uncertainty vs Prediction Error")
axs[2].legend(); axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/comparison.png", dpi=150)
print(f"\nSaved to {out_dir}/comparison.png")
print("Done!")
