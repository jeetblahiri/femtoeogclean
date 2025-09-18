from torch import nn
import torch
from torch.nn import functional as F
from utils import rel_bandpowers, alpha_peak_hz, multiband_losses, asym_band_losses

class DWConv1d(nn.Module):
    def __init__(self, cin, cout, k=7, d=1, g=4, groups=24):
        super().__init__()
        pad = (k//2)*d
        self.dw = nn.Conv1d(cin, cin, k, padding=pad, dilation=d, groups=groups, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(g, cout), num_channels=cout)
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.gn(self.pw(self.dw(x))))

class bNormEncoder(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1, bias=False), nn.BatchNorm1d(ch), nn.SiLU(),
            DWConv1d(ch, ch, 7, groups=1), DWConv1d(ch, ch, 7, d=2, groups=1))
        self.mid  = nn.Sequential(DWConv1d(ch, ch, 7, d=4, groups=1), DWConv1d(ch, ch, 7, d=8, groups=1))
        self.out  = nn.Conv1d(ch, ch, 1)
    def forward(self, x):
        # x: [B,1,512]
        h = self.out(self.mid(self.stem(x)))
        return h
    
class Encoder(nn.Module):
    def __init__(self, ch=32, groups=24, scale_up=1):
        print('Encoder using groups:', groups)
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1, bias=False), nn.GroupNorm(min(4, ch), ch), nn.SiLU(),
            DWConv1d(ch, ch, 7, groups=groups), DWConv1d(ch, ch, 7, d=2, groups=max(groups//2, 1)))
        self.mid  = nn.Sequential(DWConv1d(ch, ch, 7, d=4, groups=1), DWConv1d(ch, ch, 7, d=8, groups=1))
        self.out  = nn.Conv1d(ch, ch, 1)
    def forward(self, x):
        # x: [B,1,512]
        h = self.out(self.mid(self.stem(x)))
        return h

class shortEncoder4Conv(nn.Module):
    def __init__(self, ch=32, groups=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1, bias=False), nn.GroupNorm(min(4, ch), ch), nn.SiLU(),
            DWConv1d(ch, ch, 7, groups=groups))
        self.mid  = nn.Sequential(DWConv1d(ch, ch, 7, d=2, groups=max(groups//2, 1)), DWConv1d(ch, ch, 7, d=4, groups=1))
        self.out  = nn.Conv1d(ch, ch, 1)
    def forward(self, x):
        # x: [B,1,512]
        h = self.out(self.mid(self.stem(x)))
        return h

class shortEncoder3Conv(nn.Module):
    def __init__(self, ch=32, groups=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1, bias=False), nn.GroupNorm(min(4, ch), ch), nn.SiLU(),
            DWConv1d(ch, ch, 7, groups=groups))
        self.mid  = DWConv1d(ch, ch, 7, d=2, groups=max(groups//2, 1))
        self.out  = nn.Conv1d(ch, ch, 1, groups=1)
    def forward(self, x):
        # x: [B,1,512]
        h = self.out(self.mid(self.stem(x)))
        return h

class deeperEncoder(nn.Module):
    def __init__(self, ch=32, groups=24, scale_up=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1, bias=False), nn.GroupNorm(min(4, ch), ch), nn.SiLU(),
            DWConv1d(ch, scale_up*ch, 7, groups=max(groups//2, 1)), DWConv1d(scale_up*ch, scale_up*ch, 7, d=2, groups=groups), DWConv1d(scale_up*ch, scale_up*ch, 7, d=2, groups=groups))
        self.mid  = nn.Sequential(DWConv1d(scale_up*ch, ch, 7, d=4, groups=max(groups//2, 1)), DWConv1d(ch, ch, 7, d=8, groups=1))
        self.out  = nn.Conv1d(ch, ch, 1)
    def forward(self, x):
        # x: [B,1,512]
        h = self.out(self.mid(self.stem(x)))
        return h

class shorterHead(nn.Module):
    def __init__(self, ch=32, groups=24, scale_up=1):
        super().__init__()
        self.f_z  = nn.Conv1d(ch, ch, 1)  # head-specific bottleneck (used for independence)
        self.dec  = nn.Sequential(DWConv1d(ch, scale_up*ch, 7, groups=groups), nn.Conv1d(scale_up*ch, 1, 1))
    
    def forward(self, h):
        z = self.f_z(h)                     # [B,ch,L]
        y = self.dec(z)                     # [B,1,L]
        z_g = torch.mean(z, dim=-1)         # [B,ch] global avg for independence
        return y, z_g

class Head(nn.Module):
    def __init__(self, ch=32, groups=24, scale_up=1):
        super().__init__()
        print(ch, scale_up*ch, groups, groups//2)
        self.f_z  = nn.Conv1d(ch, ch, 1)  # head-specific bottleneck (used for independence)
        self.dec  = nn.Sequential(DWConv1d(ch, scale_up*ch, 7, d=2, groups=max(groups//2, 1)), DWConv1d(scale_up*ch, ch, 7, groups=max(groups//2, 1)), nn.Conv1d(ch, 1, 1))
    def forward(self, h):
        z = self.f_z(h)                     # [B,ch,L]
        y = self.dec(z)                     # [B,1,L]
        z_g = torch.mean(z, dim=-1)         # [B,ch] global avg for independence
        return y, z_g

class deeperHead(nn.Module):
    def __init__(self, ch=32, groups=24, scale_up=2):
        super().__init__()
        self.f_z  = nn.Conv1d(ch, ch, 1)  # head-specific bottleneck (used for independence)
        self.dec  = nn.Sequential(DWConv1d(ch, scale_up*ch, 7, d=2, groups=max(groups//2, 1)), DWConv1d(scale_up*ch, scale_up*ch, groups=groups), DWConv1d(scale_up*ch, scale_up*ch, groups=groups), DWConv1d(scale_up*ch, ch, 7, groups=max(groups//2, 1)), DWConv1d(ch, ch, 7, groups=1), nn.Conv1d(ch, 1, 1))
    def forward(self, h):
        z = self.f_z(h)                     # [B,ch,L]
        y = self.dec(z)                     # [B,1,L]
        z_g = torch.mean(z, dim=-1)         # [B,ch] global avg for independence
        return y, z_g

class shorterSingleHeadNet(nn.Module):
    def __init__(self, ch=32, device='cuda', groups=24, scale_up=1, convLevel=4):
        super().__init__()
        if convLevel==3:
            self.enc = shortEncoder3Conv(ch, groups=groups)
        elif convLevel==4:
            self.enc = shortEncoder4Conv(ch, groups=groups)
        else:
            raise ValueError("convLevel not implemented")
        self.head_s = shorterHead(ch, groups=groups, scale_up=scale_up)   # clean
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    def loss(self, batch, out, weights):
        return TwoHeadNet.loss(self, batch, out, weights)

class conv3SingleHeadNet(nn.Module):
    def __init__(self, ch=32, device='cuda', groups=24, scale_up=1, convLevel=4):
        super().__init__()
        self.enc = shortEncoder3Conv(ch, groups=groups)
        self.head_s = shorterHead(ch, groups=groups, scale_up=scale_up)   # clean
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    def loss(self, batch, out, weights):
        return TwoHeadNet.loss(self, batch, out, weights)

class conv4SingleHeadNet(nn.Module):
    def __init__(self, ch=32, device='cuda', groups=24, scale_up=1, convLevel=4):
        super().__init__()
        self.enc = shortEncoder4Conv(ch, groups=groups)
        self.head_s = shorterHead(ch, groups=groups, scale_up=scale_up)   # clean
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    def loss(self, batch, out, weights):
        return TwoHeadNet.loss(self, batch, out, weights)

class conv6SingleHeadNet(nn.Module):  
    def __init__(self, ch=32, device='cuda', groups=24, scale_up=2):
        super().__init__()
        self.enc = deeperEncoder(ch, groups=groups, scale_up=scale_up)
        self.head_s = deeperHead(ch, groups=groups, scale_up=scale_up)   # clean
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    def loss(self, batch, out, weights):
        return TwoHeadNet.loss(self, batch, out, weights)

class conv5SingleHeadNet(nn.Module):
    """Same multi-loss but single head (A1.5 in doc)"""
    
    def __init__(self, ch=32, device='cuda', groups=None, scale_up=1):
        super().__init__()
        if not groups:
            groups = ch
        self.enc = Encoder(ch, groups=groups, scale_up=scale_up)
        self.head_s = Head(ch, groups=groups, scale_up=scale_up)   # clean
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    def loss(self, batch, out, weights):
        return TwoHeadNet.loss(self, batch, out, weights)

class TwoHeadNet(nn.Module):
    def __init__(self, ch=32, device='cuda'):
        super().__init__()
        self.enc = Encoder(ch)
        self.head_s = Head(ch)   # clean
        self.head_a = Head(ch)   # artifact
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        y_a, z_a = self.head_a(h)
        return y_s, y_a, z_s, z_a

    def loss(self, batch, out, weights):
        # x = batch["x"].to(self.device)[:,None,:]
        s = batch["s"].to(self.device)[:,None,:]
        # a = batch["a"].to(self.device)[:,None,:]
        y_s, _, _, _ = out

        L_clean = F.mse_loss(y_s, s)        
        L = (weights.get("L_clean", 0)*L_clean)

        return L, dict(L_clean=float(L_clean.item()))


class bNormTwoHeadNet(TwoHeadNet):
    def __init__(self, ch=32, device='cuda'):
        super().__init__()
        self.enc = bNormEncoder(ch)
        self.head_s = Head(ch)   # clean
        self.head_a = Head(ch)   # artifact
        self.device = device

def n_params(m): 
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class baselineModel(nn.Module):
    """Single-head, MSE only (no artifact head, no additivity, no physiology, no consistency)."""
    def __init__(self, device='cuda', ch=32):
        super().__init__()
        self.enc = Encoder(ch)
        self.head_s = Head(ch)   # clean
        self.device=device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    
    def loss(self, batch, out, weights):
        y_s, y_a, z_s, z_a = out
        x = batch["x"].to(self.device)[:,None,:]
        s = batch["s"].to(self.device)[:,None,:]
        a = batch["a"].to(self.device)[:,None,:]
        
        rec_loss = F.mse_loss(y_s, s)
        
        total_loss = rec_loss
        
        return total_loss, {"L_clean":float(rec_loss.item())}
