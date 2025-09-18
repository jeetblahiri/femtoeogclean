import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import re

def normalize_key(key: str) -> str:
    key = key.lower()
    key = re.sub(r"[^\w]+", "_", key)
    key = re.sub(r"_+", "_", key)
    return key.strip("_")

def parse_metrics_file(path):
    with open(path) as f:
        text = f.read()
    rrmse_match = re.search(r"RRMSE.*?([-+]?\d*\.\d+|\d+)", text)
    metrics_json = re.search(r"\{.*\}", text, re.S)
    metrics = json.loads(metrics_json.group(0)) if metrics_json else {}
    return {
        "rrmse": float(rrmse_match.group(1)) if rrmse_match else None,
        "rmse": metrics.get("RMSE"),
        "corr": metrics.get("Corr"),
        "sdr": metrics.get("SDR"),
    }
'''
def parse_metrics_file(path):
    results = {}
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # JSON block
        if line.startswith("{"):
            obj_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("}"):
                obj_lines.append(lines[i])
                i += 1
            obj_lines.append(lines[i])  # closing brace
            metrics = json.loads("\n".join(obj_lines))
            for k, v in metrics.items():
                try:
                    results[normalize_key(k)] = float(v)
                except (ValueError, TypeError):
                    results[normalize_key(k)] = np.nan

        else:
            # Handle "key: value" or "key - value"
            if ":" in line:
                key, val = line.split(":", 1)
            elif "-" in line:
                key, val = line.split("-", 1)
            else:
                i += 1
                continue

            val = val.strip()
            try:
                results[normalize_key(key)] = float(val)
            except ValueError:
                results[normalize_key(key)] = np.nan

        i += 1

    # Apply renames
    rename_map = {
        "rrmse_temporal": "rrmse",
        "alphashift_hz_median": "alpha",
        "artifact_reconstruction_l1_lower_is_better": "l1_rec",
        "additivity_l1_x_s_a": "add",
    }
    for old, new in rename_map.items():
        if old in results:
            results[new] = results.pop(old)

    return results
'''
def rms(x, axis=-1):
    return np.sqrt(np.mean(x**2, axis=axis) + 1e-12)

def band_indices(n, fs, lo, hi, device):
    freqs = torch.linspace(0, fs/2, n//2+1, device=device)
    return (freqs >= lo) & (freqs < hi)

BANDS = {"delta":(0.5,4), "theta":(4,8), "alpha":(8,12), "beta":(12,30), "gamma":(30,40)}

def rel_bandpowers(x, fs=256):
    """
    total power in each band, normalized for 0.5-40 Hz
    could make it average power to reflect different band widths?
    """
    # x: [B,1,L]
    X = torch.fft.rfft(x, dim=-1)
    psd = (X.real**2 + X.imag**2) / x.shape[-1]  # power
    freqs = torch.linspace(0, fs/2, x.shape[-1]//2+1, device=x.device)
    m = (freqs >= 0.5) & (freqs <= 40.0)
    total = psd[..., m].sum(dim=-1, keepdim=True) + 1e-9
    outs = []
    for (lo,hi) in BANDS.values():
        sel = (freqs >= lo) & (freqs < hi)
        outs.append( (psd[..., sel].sum(dim=-1, keepdim=True) / total) )
    return torch.cat(outs, dim=-1)  # [B,1,5] -> we'll squeeze

# def alpha_peak_hz(x, fs=256):
#     X = torch.fft.rfft(x, dim=-1)
#     psd = (X.real**2 + X.imag**2) / x.shape[-1]
#     freqs = torch.linspace(0, fs/2, x.shape[-1]//2+1, device=x.device)
#     sel = (freqs >= 8.0) & (freqs <= 12.0)
#     idx = torch.argmax(psd[..., sel], dim=-1)
#     f_alpha = freqs[sel][idx]
#     return f_alpha  # [B,1]

def alpha_peak_hz(x, fs=256, temp=1.0):
    X = torch.fft.rfft(x, dim=-1)
    psd = (X.real**2 + X.imag**2) / x.shape[-1]
    freqs = torch.linspace(0, fs/2, x.shape[-1]//2+1, device=x.device)

    sel = (freqs >= 8.0) & (freqs <= 12.0)
    psd_sel = psd[..., sel]
    freqs_sel = freqs[sel]

    # softmax over PSD as weights
    weights = torch.softmax(psd_sel / temp, dim=-1)
    f_alpha = (weights * freqs_sel).sum(dim=-1)
    return f_alpha

def asym_band_losses(s_hat, s, x=None, fs=256, nfft=512, tau=0.2,
                     kedge=6.0, margin_bg=0.02):
    """
    Asymmetric physiology losses:
      - L_dta: symmetric RP for delta, theta, alpha (|RP_hat - RP_true|)
      - L_bg_hi: one-sided hinge for beta/gamma overshoot (max(RP_hat - RP_true - m, 0))
      - L_alpha_cent: alpha centroid |f*_alpha(hat) - f*_alpha(true)|
    Returns: L_dta, L_bg_hi, L_alpha_cent
    """
    freqs, P_sh = psd_rfft(s_hat, fs=fs, nfft=nfft, smooth=5)
    _,     P_s  = psd_rfft(s,     fs=fs, nfft=nfft, smooth=5)

    BANDS = {"delta": (0.5,4.0), "theta": (4.0,8.0),
             "alpha": (8.0,12.0), "beta": (12.0,30.0), "gamma": (30.0,45.0)}

    # Relative powers (scale-robust)
    RP_sh = relative_powers(P_sh, freqs, bands=BANDS, kedge=kedge)  # (B,5)
    RP_s  = relative_powers(P_s,  freqs, bands=BANDS, kedge=kedge)

    # Indices
    idx = {"delta":0, "theta":1, "alpha":2, "beta":3, "gamma":4}
    dta = [idx["delta"], idx["theta"], idx["alpha"]]
    bg  = [idx["beta"], idx["gamma"]]

    # (1) Symmetric for delta/theta/alpha
    L_dta = (RP_sh[:, dta] - RP_s[:, dta]).abs().sum(dim=-1).mean()

    # (2) One-sided for beta/gamma: penalise only overshoot vs clean + margin
    overshoot = (RP_sh[:, bg] - RP_s[:, bg] - margin_bg).clamp(min=0.0)
    L_bg_hi = overshoot.sum(dim=-1).mean()

    # (3) Alpha centroid (soft-argmax in alpha)
    f1, f2 = BANDS["alpha"]
    c_sh = band_centroid(P_sh, freqs, f1, f2, tau=tau, kedge=kedge)  # (B,)
    c_s  = band_centroid(P_s,  freqs, f1, f2, tau=tau, kedge=kedge)
    L_alpha_cent = (c_sh - c_s).abs().mean()

    # print(L_dta.mean().item(), L_bg_hi.mean().item(), L_alpha_cent.mean().item())

    return L_dta, L_bg_hi, L_alpha_cent

# def loss_block(batch, out, weights):
#     x = batch["x"].to(device)[:,None,:]
#     s = batch["s"].to(device)[:,None,:]
#     a = batch["a"].to(device)[:,None,:]
#     y_s, y_a, z_s, z_a = out

#     L_clean = F.mse_loss(y_s, s)
#     L_art   = F.mse_loss(y_a, a)
#     L_sum   = F.l1_loss(x, y_s + y_a)

#     # head-independence (cosine ~ 0)
#     cs = F.cosine_similarity(z_s, z_a, dim=1)
#     L_ind = cs.abs().mean()

#     # physiology
#     rb_s  = rel_bandpowers(s).squeeze(-2)   # [B,5]
#     rb_sh = rel_bandpowers(y_s).squeeze(-2)
#     L_bands = F.l1_loss(rb_sh, rb_s)

#     f_s   = alpha_peak_hz(s)
#     f_sh  = alpha_peak_hz(y_s)
#     L_alpha = (f_sh - f_s).abs().mean()

#     # optional: same-clean consistency on encoder features (view x2)
#     L_cons = torch.tensor(0., device=device)
#     if "x2" in batch:
#         x2 = batch["x2"].to(device)[:,None,:]
#         with torch.no_grad():
#             h = model.enc(x)
#         h2 = model.enc(x2)
#         z1 = torch.mean(h,  dim=-1)
#         z2 = torch.mean(h2, dim=-1)
#         z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
#         L_cons = ((z1 - z2)**2).mean()

#     w = weights
#     L = (L_clean 
#          + w["art"]*L_art 
#          + w["sum"]*L_sum 
#          + w["ind"]*L_ind 
#          + w["bands"]*L_bands 
#          + w["alpha"]*L_alpha 
#          + w["cons"]*L_cons)
#     return L, dict(L_clean=float(L_clean.item()),
#                    L_art=float(L_art.item()), L_sum=float(L_sum.item()),
#                    L_ind=float(L_ind.item()), L_bands=float(L_bands.item()),
#                    L_alpha=float(L_alpha.item()), L_cons=float(L_cons.item()))

def plot_loss(tl, vl, path):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(tl, 'b-', label="Train Loss")
    ax2.plot(vl, 'r-', label="Val Loss")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color='b')
    ax2.set_ylabel("Val Loss", color='r')

    # save
    plt.title("Training and Validation Loss")
    fig.tight_layout()
    plt.savefig(path)
    plt.close()

def mse(pred, target):
    return np.mean((pred - target) ** 2)

def rrmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2)) / np.sqrt(np.mean(target**2))

def relative(pred, target):
    return np.mean(np.abs(pred - target) / np.abs(target))

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def corr(a,b):
    a0=a-a.mean(); b0=b-b.mean()
    return float(np.sum(a0*b0)/(np.linalg.norm(a0)*np.linalg.norm(b0)+1e-12))
def sdr(y, s):
    num = np.sum(s**2); den = np.sum((s-y)**2)+1e-12
    return float(10*np.log10(num/den))

def metrics_on_pair(Y, S, fs=256):
    # Y,S: [N, 512] numpy

    out = {
        "RRMSE": rrmse(Y,S),
        "RMSE": rmse(Y,S),
        "Corr": corr(Y,S),
        "SDR":  sdr(Y,S)
        # "RRMSE": rrmse(Y,S),
        # "Relative": relative(Y,S)
    }
    # print(f'RRMSE - {rrmse(Y,S)}')
    # band corr & alpha shift
    def np_psd(x):
        X = np.fft.rfft(x); psd = (X.real**2 + X.imag**2)/x.shape[-1]
        freqs = np.linspace(0, fs/2, x.shape[-1]//2+1)
        return psd, freqs
    psdY, f = np_psd(Y); psdS, _ = np_psd(S)
    band = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,12),"beta":(12,30),"gamma":(30,40)}
    for k,(lo,hi) in band.items():
        sel = (f>=lo)&(f<hi)
        by = psdY[..., sel].sum(-1); bs = psdS[..., sel].sum(-1)
        out[f"Corr_{k}"] = np.corrcoef(by, bs)[0,1]
    sel = (f>=8)&(f<=12)
    fa = f[sel][np.argmax(psdS[..., sel], axis=-1)]
    fay = f[sel][np.argmax(psdY[..., sel], axis=-1)]
    out["AlphaShift(Hz)_median"] = float(np.median(np.abs(fay-fa)))
    return out

def save_metrics(path, model, ds_tst, X_te, S_te, weights=None):
    model.eval()
    Ys=[]; As=[]; Ss=[]
    with torch.no_grad():
        for batch in ds_tst:
            x = batch["x"].to(model.device)[:,None,:]
            s = batch["s"].to(model.device)[:,None,:]
            y_s, y_a, _, _ = model(x)

            if y_a is not None:
                Ys.append(y_s.squeeze(1).cpu().numpy())
                As.append(y_a.squeeze(1).cpu().numpy())
                Ss.append(s.squeeze(1).cpu().numpy())

            else:
                Ys.append(y_s.squeeze(1).cpu().numpy())
                Ss.append(s.squeeze(1).cpu().numpy())

        if y_a is not None:
                Y = np.concatenate(Ys,0); Ahat = np.concatenate(As,0); S = np.concatenate(Ss,0)
                print(X_te.shape, S_te.shape, Y.shape, Ahat.shape)
                add_err = np.mean(np.abs((X_te - S_te) - Ahat))
                rec_err = np.mean(np.abs(X_te - (Y + Ahat)))
        else:
            Y = np.concatenate(Ys,0); S = np.concatenate(Ss,0)
            add_err = None
            rec_err = None
        

        with open(path, "w") as f:
            if weights is not None:
                f.write('Loss weights-\n')
                for k, v in weights.items():
                    f.write(f'{k} - {v} | ')
            f.write(f"RRMSE (temporal) - {rrmse(Y, S)}\n")
            f.write(json.dumps(metrics_on_pair(Y, S, fs=256), indent=2) + "\n")
            f.write(f"Artifact reconstruction L1 (lower is better): {add_err}\n")
            f.write(f"Additivity L1 |x - (ŝ+â)|: {rec_err}\n")

import torch
import torch.nn.functional as F

# --------- frequency grid & soft masks ----------
def make_freq_grid(nfft, fs):
    # rfft bins: 0..fs/2 inclusive
    freqs = torch.linspace(0., fs/2, nfft//2 + 1, device='cpu')  # move to device later
    return freqs

def soft_band_mask(freqs, f1, f2, k=4.0):
    # smooth box: σ(k(f-f1)) - σ(k(f-f2))
    return torch.sigmoid(k*(freqs - f1)) - torch.sigmoid(k*(freqs - f2))

def eeg_pass_mask(freqs, f_lo=0.5, f_hi=45.0, k=4.0):
    return soft_band_mask(freqs, f_lo, f_hi, k)

# --------- PSD (differentiable) ----------
def psd_rfft(x, fs=256, nfft=512, smooth=5):
    """
    x: (B, T) time-domain signal in torch
    Returns freqs (1,F), P (B,F) with Hann window + mild freq smoothing.
    """
    device = x.device
    T = x.shape[-1]
    if nfft != T:
        # zero-pad or trim (keeps autograd)
        if nfft > T:
            pad = (0, nfft - T)
            xz = F.pad(x, pad)
        else:
            xz = x[..., :nfft]
    else:
        xz = x
    # Hann window
    n = torch.arange(nfft, device=device).float()
    w = 0.5 - 0.5*torch.cos(2*torch.pi*n/(nfft-1))
    xw = xz * w  # (B,nfft)
    # RFFT
    X = torch.fft.rfft(xw, n=nfft, dim=-1)  # (B, F)
    P = (X.real*2 + X.imag*2) / (w.pow(2).sum() + 1e-9)  # periodogram
    # Frequency smoothing: 1D conv with fixed kernel
    if smooth > 1:
        k = torch.ones(1,1,smooth, device=device) / smooth
        P = F.pad(P.unsqueeze(1), (smooth//2, smooth//2), mode='replicate')
        P = F.conv1d(P, k).squeeze(1)
    freqs = torch.linspace(0., fs/2, P.shape[-1], device=device).unsqueeze(0)  # (1,F)
    return freqs, P + 1e-12  # eps for stability

# --------- band features ----------
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (12.0, 30.0),
    "gamma": (30.0, 45.0),   # keep <=45Hz at 256Hz fs to avoid line noise
}

def relative_powers(P, freqs, bands=BANDS, kedge=6.0, eeg_lo=0.5, eeg_hi=45.0):
    m_tot = eeg_pass_mask(freqs, eeg_lo, eeg_hi, k=kedge)  # (1,F)
    denom = (m_tot * P).sum(dim=-1, keepdim=True) + 1e-12  # (B,1)
    rps = []
    for (f1,f2) in bands.values():
        mb = soft_band_mask(freqs, f1, f2, k=kedge)        # (1,F)
        num = (mb * P).sum(dim=-1, keepdim=True)           # (B,1)
        rps.append(num / denom)
    return torch.cat(rps, dim=-1)  # (B, nbands)

def band_centroid(P, freqs, f1, f2, tau=0.2, kedge=6.0):
    mb = soft_band_mask(freqs, f1, f2, k=kedge)            # (1,F)
    W = torch.exp((P) / tau) * mb                          # (B,F)
    num = (W * freqs).sum(dim=-1)
    den = W.sum(dim=-1) + 1e-12
    return num / den                                       # (B,)

def multiband_losses(s_hat, s, fs=256, nfft=512, tau=0.2,
                     w_rp=1.0, w_cent=1.0, bands=BANDS):
    """
    Returns L_RP, L_cent, and per-band dicts (useful for logging).
    """
    freqs, P_sh = psd_rfft(s_hat, fs=fs, nfft=nfft, smooth=5)
    _,     P_s  = psd_rfft(s,     fs=fs, nfft=nfft, smooth=5)
    # Ensure freqs shape broadcast
    # Relative power vector
    RP_sh = relative_powers(P_sh, freqs, bands=bands)  # (B, nb)
    RP_s  = relative_powers(P_s,  freqs, bands=bands)
    L_RP  = (RP_sh - RP_s).abs().sum(dim=-1).mean()

    # Centroids per band
    L_cent = 0.0
    centroids = {}
    for name,(f1,f2) in bands.items():
        c_sh = band_centroid(P_sh, freqs, f1, f2, tau=tau)
        c_s  = band_centroid(P_s,  freqs, f1, f2, tau=tau)
        centroids[name] = (c_sh.mean().item(), c_s.mean().item())
        L_cent = L_cent + (c_sh - c_s).abs().mean()

    return w_rp*L_RP, w_cent*L_cent, {"RP": RP_sh.mean(0).detach().cpu().tolist(),
                                      "centroids": centroids}