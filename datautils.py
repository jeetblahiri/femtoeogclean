import numpy as np

def load_npy_safely(path, N_SAMPLES, mmap=False, znorm=False):
    """Load .npy as memmap (read-only) and coerce shape to (N, 512)."""
    arr = np.load(path, mmap_mode='r' if mmap else None)

    # Ensure ndarray view (memmap is fine)
    a = np.asarray(arr)

    # Handle common layouts: (N,1,512), (N,512,1), (1,N,512), (N,512), (512,N)
    if a.ndim == 1:
        a = a[None, :]                               # (1, 512)
    elif a.ndim == 2:
        if a.shape[1] == N_SAMPLES:                  # (N, 512) OK
            pass
        elif a.shape[0] == N_SAMPLES:                # (512, N) -> transpose
            a = a.T
        else:
            raise AssertionError(f"Unexpected 2D shape {a.shape}, can't find length=512.")
    elif a.ndim == 3:
        # Move the 512-length axis to the last dim
        axes_with_512 = [i for i, s in enumerate(a.shape) if s == N_SAMPLES]
        if not axes_with_512:
            raise AssertionError(f"No axis of length {N_SAMPLES} in shape {a.shape}.")
        l_ax = axes_with_512[-1]
        if l_ax != a.ndim - 1:
            a = np.moveaxis(a, l_ax, -1)            # (..., 512) at the end

        # Now squeeze any singleton "channel" axis to get (N, 512)
        if a.ndim == 3 and (a.shape[1] == 1):
            a = a[:, 0, :]                           # (N, 512)
        elif a.ndim == 3 and (a.shape[0] == 1):
            a = a[0, :, :]                           # (N, 512)
        elif a.ndim == 3 and (a.shape[-2] == 1):
            a = a.reshape(a.shape[0], a.shape[-1])   # (N, 512)
        elif a.ndim == 3:
            raise AssertionError(f"3D shape still ambiguous after moveaxis: {a.shape}")
    else:
        raise AssertionError(f"Unsupported ndim={a.ndim} for shape {a.shape}")

    assert a.ndim == 2 and a.shape[1] == N_SAMPLES, f"Expected (N, {N_SAMPLES}), got {a.shape}"

    if znorm:
        # z-score normalise
        a = (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-9)

    return a  # still memmap-backed if mmap=True

# === Build datasets & dataloaders ===
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Uses the memmap-aware arrays you already loaded: X_tr, S_tr, X_te, S_te
class EEGPairs(Dataset):
    def __init__(self, X_memmap, S_memmap, make_view=True):
        self.X = X_memmap.astype(np.float32)  # (N, 512)
        self.S = S_memmap.astype(np.float32)  # (N, 512)
        self.make_view = make_view

    def __len__(self): return self.X.shape[0]

    def _remix(self, s, a):
        g = float(torch.empty(1).uniform_(0.7, 1.3))
        a2 = a * g
        if torch.rand(1).item() < 0.5:
            scale = float(torch.empty(1).uniform_(0.95, 1.05))
            L = a2.shape[-1]
            a2r = torch.nn.functional.interpolate(a2[None, None, :], size=int(L*scale),
                                                  mode="linear", align_corners=False)[0,0]
            a2 = a2r[:L] if a2r.numel() >= L else torch.nn.functional.pad(a2r, (0, L-a2r.numel()))
        return s + a2

    def __getitem__(self, idx):
        import numpy as np
        x = torch.from_numpy(self.X[idx])
        s = torch.from_numpy(self.S[idx])
        a = x - s
        sample = {"x": x, "s": s, "a": a}
        # if self.make_view:
        #     sample["x2"] = self._remix(s, a)
        return sample
    
def make_data(X_tr, S_tr, X_te, S_te, frac=0.1):
    # Datasets
    ds_tr = EEGPairs(X_tr, S_tr, make_view=True)
    ds_te = EEGPairs(X_te, S_te, make_view=False)

    # Train/val split
    N = len(ds_tr)
    n_val = max(1, int(0.1 * N))
    n_train = N - n_val
    g = torch.Generator().manual_seed(0)
    ds_trn, ds_val = random_split(ds_tr, [n_train, n_val], generator=g)

    # DataLoaders (tune num_workers to your box; 0 is safest)
    batch_size = 128
    dl_trn = DataLoader(ds_trn, batch_size=batch_size, shuffle=True,  drop_last=True,
                        num_workers=2, pin_memory=torch.cuda.is_available())
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    dl_tst = DataLoader(ds_te,  batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    # Smoke test: fetch one batch and check shapes
    b = next(iter(dl_trn))
    print({k: v.shape for k, v in b.items()})

    return dl_trn, dl_val, dl_tst