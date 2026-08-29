"""PGSRNet: prior-gated adaptation with two-stage structural refinement."""

from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.infMAE_uperhead import (
    MaskedAutoencoderInfMAE,
    load_infmae_pretrain,
    UPerHead,
    InfMAEBackbone,
)
from models.patch_unet import DualBranchLiteUNetV2
# ---------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------
def init_kaiming(module: torch.nn.Module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def _meshgrid_ij(y, x):
    try:
        return torch.meshgrid(y, x, indexing="ij")
    except TypeError:
        return torch.meshgrid(y, x)


# ---------------------------------------------------------
# ✅ Multi-task uncertainty weighting wrapper (Kendall-style)
# ---------------------------------------------------------
class MultiTaskLossWrapper(nn.Module):
    """
    mtLoss: list[Tensor] length == task_num
    loss_sum = Σ ( 0.5/σ_i^2 * L_i + log(1+σ_i^2) ), where σ_i is learnable (params[i])
    """
    def __init__(self, task_num: int):
        super().__init__()
        self.task_num = int(task_num)
        self.params = nn.Parameter(torch.ones(self.task_num))

    def forward(self, mtLoss):
        if not isinstance(mtLoss, (list, tuple)) or len(mtLoss) != self.task_num:
            raise ValueError(
                f"mtLoss must be list/tuple with length={self.task_num}, got {type(mtLoss)} "
                f"len={len(mtLoss) if isinstance(mtLoss,(list,tuple)) else 'NA'}"
            )
        loss_sum = 0
        for i, loss in enumerate(mtLoss):
            loss_sum = loss_sum + 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


# ---------------------------------------------------------
# Loss utils
# ---------------------------------------------------------
def _to_mask_4d(masks: torch.Tensor, device=None):
    """
    masks: [B,H,W] or [B,1,H,W]
    return: [B,1,H,W] float in {0,1}
    """
    if device is None:
        device = masks.device
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    masks = (masks > 0).float().to(device)
    return masks


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)).clamp_min(eps)
    dice = 1.0 - (num + eps) / (den + eps)
    return dice.mean()


def seg_bce_dice_loss(logits: torch.Tensor, gt: torch.Tensor, pos_weight: float = 9.0):
    """
    logits: [B,1,H,W]
    gt:     [B,1,H,W]
    """
    device = logits.device
    pw = torch.tensor([pos_weight], device=device)
    bce = F.binary_cross_entropy_with_logits(
        logits.squeeze(1),
        gt.squeeze(1),
        pos_weight=pw,
    )
    dsc = dice_loss_with_logits(logits, gt)
    return bce + dsc

def seg_bce_dice_fp_loss(
    logits: torch.Tensor,
    gt: torch.Tensor,
    pos_weight: float = 9.0,
    lambda_fp: float = 0.5,
    fp_power: float = 2.0,
):
    """
    logits: [B,1,H,W]
    gt:     [B,1,H,W]

    Loss = BCE + Dice + lambda_fp * FP_penalty

    FP_penalty suppresses false alarms in background regions:
        FP = prob^p * (1 - gt)
    """
    device = logits.device
    gt = gt.float()

    if gt.ndim == 3:
        gt = gt.unsqueeze(1)

    pw = torch.tensor([pos_weight], device=device)

    bce = F.binary_cross_entropy_with_logits(
        logits.squeeze(1),
        gt.squeeze(1),
        pos_weight=pw,
    )

    dsc = dice_loss_with_logits(logits, gt)

    prob = torch.sigmoid(logits)
    bg = 1.0 - gt

    fp_loss = ((prob ** fp_power) * bg).mean()

    return bce + dsc + lambda_fp * fp_loss

# ---------------------------------------------------------
# Sobel / edge (used in Stage2 patch input and PGRA prior gate)
# ---------------------------------------------------------
class SobelHF(nn.Module):
    """Fixed Sobel edge magnitude for single-channel maps."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x_1chw):
        kx = self.kx.to(dtype=x_1chw.dtype)
        ky = self.ky.to(dtype=x_1chw.dtype)
        gx = F.conv2d(x_1chw, kx, padding=1)
        gy = F.conv2d(x_1chw, ky, padding=1)
        g = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return g


class _Norm2d(nn.Module):
    def __init__(self, c: int, norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        norm = norm.lower()
        if norm == "bn":
            self.norm = nn.BatchNorm2d(c)
        elif norm == "gn":
            g = min(int(gn_groups), int(c))
            while c % g != 0 and g > 1:
                g -= 1
            self.norm = nn.GroupNorm(g, c)
        else:
            raise ValueError(f"Unknown norm: {norm}")

    def forward(self, x):
        return self.norm(x)


# ---------------------------------------------------------
# ✅ PGRA: Prior-Gated Residual Adapters (Token / 2D)
# ---------------------------------------------------------
class PriorGate2D(nn.Module):
    """Additive prior gate for 2D features: sigmoid(wc*C + we*E - wb*B)."""

    def __init__(self, c: int, k_stat: int = 3):
        super().__init__()
        self.k = int(k_stat)

        self.p_dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.p_out = nn.Conv2d(c, 1, 1, bias=True)

        self.b_out = nn.Conv2d(1, 1, 1, bias=True)

        self.proj = nn.Conv2d(c, 1, 1, bias=False)
        self.sobel = SobelHF()
        self.e_out = nn.Conv2d(1, 1, 1, bias=True)

        self.wp = nn.Parameter(torch.tensor(1.0))
        self.wb = nn.Parameter(torch.tensor(1.0))
        self.we = nn.Parameter(torch.tensor(1.0))

        self.apply(init_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool2d(x, kernel_size=self.k, stride=1, padding=self.k // 2)

        hf = x - smooth
        P = torch.sigmoid(self.p_out(self.p_dw(hf)))  # target-enhancement / local contrast prior

        ix = x.mean(dim=1, keepdim=True)
        mu = F.avg_pool2d(ix, kernel_size=self.k, stride=1, padding=self.k // 2)
        mu2 = F.avg_pool2d(ix * ix, kernel_size=self.k, stride=1, padding=self.k // 2)
        var = (mu2 - mu * mu).clamp_min(0.0)
        Bm = torch.sigmoid(self.b_out(var))           # clutter-complexity prior

        E = torch.sigmoid(self.e_out(self.sobel(self.proj(x))))  # structure / edge prior

        return torch.sigmoid(self.wp * P + self.we * E - self.wb * Bm)

def _isqrt(n: int) -> int:
    s = int(n ** 0.5)
    while (s + 1) * (s + 1) <= n:
        s += 1
    while s * s > n:
        s -= 1
    return s


class PriorGateToken(nn.Module):
    """Additive prior gate for token features."""

    def __init__(self, dim: int, k_stat: int = 3):
        super().__init__()
        self.k = int(k_stat)

        self.p_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.p_out = nn.Conv2d(dim, 1, 1, bias=True)

        self.b_out = nn.Conv2d(1, 1, 1, bias=True)

        self.proj = nn.Conv2d(dim, 1, 1, bias=False)
        self.sobel = SobelHF()
        self.e_out = nn.Conv2d(1, 1, 1, bias=True)

        self.wp = nn.Parameter(torch.tensor(1.0))
        self.wb = nn.Parameter(torch.tensor(1.0))
        self.we = nn.Parameter(torch.tensor(1.0))

        self.apply(init_kaiming)


    def forward(self, x_tok: torch.Tensor) -> torch.Tensor:
        B, N, C = x_tok.shape
        s = _isqrt(N)
        if s * s != N:
            return x_tok.new_zeros((B, N, 1))

        x = x_tok.transpose(1, 2).contiguous().view(B, C, s, s)
        smooth = F.avg_pool2d(x, kernel_size=self.k, stride=1, padding=self.k // 2)

        hf = x - smooth
        P = torch.sigmoid(self.p_out(self.p_dw(hf)))

        ix = x.mean(dim=1, keepdim=True)
        mu = F.avg_pool2d(ix, kernel_size=self.k, stride=1, padding=self.k // 2)
        mu2 = F.avg_pool2d(ix * ix, kernel_size=self.k, stride=1, padding=self.k // 2)
        var = (mu2 - mu * mu).clamp_min(0.0)
        Bm = torch.sigmoid(self.b_out(var))

        E = torch.sigmoid(self.e_out(self.sobel(self.proj(x))))

        g2 = torch.sigmoid(self.wp * P + self.we * E - self.wb * Bm)
        g = g2.flatten(2).transpose(1, 2).contiguous()
        return g


class PGRAToken(nn.Module):
    """Prior-gated token residual adapter."""
    def __init__(
        self,
        dim: int,
        r: int,
        k_stat: int = 3,
        detach_gate: bool = True,
    ):
        super().__init__()
        r = int(max(8, r))
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, r, bias=True)
        self.act = nn.GELU()
        self.up = nn.Linear(r, dim, bias=True)

        self.gate = PriorGateToken(dim, k_stat=k_stat)
        self.detach_gate = bool(detach_gate)

        self.s = nn.Parameter(torch.tensor(1.0))

        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        z = self.up(self.act(self.down(self.ln(x))))
        g = self.gate(x)
        if self.detach_gate:
            g = g.detach()
        return z * g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = 2.0 * torch.tanh(self.s)
        return x + s * self.forward_delta(x)


class PGRA2D(nn.Module):
    """Prior-gated 2D residual adapter."""
    def __init__(
        self,
        c: int,
        r: int,
        norm: str = "gn",
        gn_groups: int = 16,
        k_stat: int = 3,
        detach_gate: bool = True,
    ):
        super().__init__()
        r = int(max(8, r))
        self.norm = _Norm2d(c, norm=norm, gn_groups=gn_groups)
        self.down = nn.Conv2d(c, r, 1, bias=True)
        self.act = nn.GELU()
        self.up = nn.Conv2d(r, c, 1, bias=True)

        self.gate = PriorGate2D(c, k_stat=k_stat)
        self.detach_gate = bool(detach_gate)

        self.s = nn.Parameter(torch.tensor(1.0))

        self.apply(init_kaiming)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        z = self.up(self.act(self.down(self.norm(x))))
        g = self.gate(x)
        if self.detach_gate:
            g = g.detach()
        return z * g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = 2.0 * torch.tanh(self.s)
        return x + s * self.forward_delta(x)


# ---------------------------------------------------------
# ✅ Block wrappers: placement unchanged (attn branch + mlp branch)
# ---------------------------------------------------------
class _TokenBlockWithPGRA(nn.Module):
    """
    Same placement as your IRST version:
      - apply PGRA delta on the attn branch output u
      - apply PGRA delta on the mlp branch output v
    """
    def __init__(self, blk: nn.Module, dim: int, r_ratio: float = 0.25,
                 k_stat: int = 3, detach_gate: bool = True):
        super().__init__()
        self.blk = blk
        r = max(8, int(dim * float(r_ratio)))
        self.Space_Adapter = PGRAToken(dim=dim, r=r, k_stat=k_stat, detach_gate=detach_gate)
        self.MLP_Adapter = PGRAToken(dim=dim, r=r, k_stat=k_stat, detach_gate=detach_gate)

    def forward(self, x: torch.Tensor):
        if not all(hasattr(self.blk, k) for k in ["norm1", "attn", "norm2", "mlp"]):
            y = self.blk(x)
            return y + self.MLP_Adapter.s * self.MLP_Adapter.forward_delta(y)

        drop_path = getattr(self.blk, "drop_path", nn.Identity())

        shortcut = x
        u = self.blk.attn(self.blk.norm1(x))
        u = self.Space_Adapter(u)
        x = shortcut + drop_path(u)

        v = self.blk.mlp(self.blk.norm2(x))
        v = self.MLP_Adapter(v)
        x = x + drop_path(v)
        return x



class _CBlockWithPGRA(nn.Module):
    """
    Keep your explicit CBlock forward (mask-aware), replace adapter delta with PGRA2D.
    """
    def __init__(self, blk: nn.Module, dim: int, r_ratio: float = 0.25,
                 gn_groups: int = 16, k_stat: int = 3, detach_gate: bool = True):
        super().__init__()
        self.blk = blk
        r = max(8, int(dim * float(r_ratio)))
        self.Space_Adapter = PGRA2D(
            c=dim, r=r, norm="gn", gn_groups=gn_groups,
            k_stat=k_stat, detach_gate=detach_gate,
        )
        self.MLP_Adapter = PGRA2D(
            c=dim, r=r, norm="gn", gn_groups=gn_groups,
            k_stat=k_stat, detach_gate=detach_gate,
        )

    def forward(self, x: torch.Tensor, mask=None):
        drop_path = getattr(self.blk, "drop_path", nn.Identity())

        shortcut = x

        x1 = self.blk.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        u = self.blk.conv1(x1)
        if mask is not None:
            u = u * mask
        u = self.blk.attn(u)
        u = self.blk.conv2(u)

        u = self.Space_Adapter(u)
        x = shortcut + drop_path(u)

        x2 = self.blk.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        v = self.blk.mlp(x2)
        v = self.MLP_Adapter(v)
        x = x + drop_path(v)

        return x



# ---------------------------------------------------------
# Encoder wrapper: inject PGRA adapters (replaces IRST adapters)
# ---------------------------------------------------------
class MaskedAutoencoderInfMAE_PGRAAdapter(MaskedAutoencoderInfMAE):
    def __init__(
        self,
        *args,
        adapter_mlp_ratio_token: float = 0.25,
        adapter_mlp_ratio_2d: float = 0.25,
        adapter_gn_groups: int = 16,
        adapter_k_stat: int = 3,
        adapter_detach_gate: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        embed_dims = kwargs.get("embed_dim", None)
        if embed_dims is None and len(args) > 3:
            embed_dims = args[3]
        if embed_dims is None:
            embed_dims = [256, 384, 768]

        self._adp_cfg = dict(
            r_ratio_token=float(adapter_mlp_ratio_token),
            r_ratio_2d=float(adapter_mlp_ratio_2d),
            gn_groups=int(adapter_gn_groups),
            k_stat=int(adapter_k_stat),
            detach_gate=bool(adapter_detach_gate),
            embed_dims=list(embed_dims),
        )
        self._pgra_injected = False

    def inject_pgra_adapters(self):
        if self._pgra_injected:
            return

        cfg = self._adp_cfg
        ed = cfg["embed_dims"]
        dim1, dim2, dim3 = int(ed[0]), int(ed[1]), int(ed[2])

        if hasattr(self, "blocks1"):
            for i in range(len(self.blocks1)):
                self.blocks1[i] = _CBlockWithPGRA(
                    blk=self.blocks1[i],
                    dim=dim1,
                    r_ratio=cfg["r_ratio_2d"],
                    gn_groups=cfg["gn_groups"],
                    k_stat=cfg["k_stat"],
                    detach_gate=cfg["detach_gate"],
                )

        if hasattr(self, "blocks2"):
            for i in range(len(self.blocks2)):
                self.blocks2[i] = _CBlockWithPGRA(
                    blk=self.blocks2[i],
                    dim=dim2,
                    r_ratio=cfg["r_ratio_2d"],
                    gn_groups=cfg["gn_groups"],
                    k_stat=cfg["k_stat"],
                    detach_gate=cfg["detach_gate"],
                )

        if hasattr(self, "blocks3"):
            for i in range(len(self.blocks3)):
                self.blocks3[i] = _TokenBlockWithPGRA(
                    blk=self.blocks3[i],
                    dim=dim3,
                    r_ratio=cfg["r_ratio_token"],
                    k_stat=cfg["k_stat"],
                    detach_gate=cfg["detach_gate"],
                )

        self._pgra_injected = True


def freeze_infmae_except_pgra_adapters(encoder: nn.Module):
    for p in encoder.parameters():
        p.requires_grad = False

    for n, p in encoder.named_parameters():
        if (
            ("Space_Adapter" in n) or
            ("MLP_Adapter" in n) or
            (n.endswith(".s")) or
            (".gate." in n) or
            (n.endswith(".wp")) or (n.endswith(".wb")) or (n.endswith(".we"))
        ):
            p.requires_grad = True


# ---------------------------------------------------------
# BMSR (image-scale refinement, optional)
# ---------------------------------------------------------
class BMSRRefiner(nn.Module):
    def __init__(self, in_ch_image=3, in_ch_logits=1, mid=64, k=3, norm="gn", gn_groups=16):
        super().__init__()
        self.to_gray = nn.Conv2d(in_ch_image, 1, 1, bias=False)
        with torch.no_grad():
            self.to_gray.weight.data[:] = 1.0 / in_ch_image
        self.to_gray.requires_grad_(False)

        self.hf = SobelHF()

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch_logits + 1 + 1, mid, k, 1, k // 2, bias=False),
            _Norm2d(mid, norm=norm, gn_groups=gn_groups),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid, mid, k, 1, k // 2, bias=False),
            _Norm2d(mid, norm=norm, gn_groups=gn_groups),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid, mid, k, 1, k // 2, bias=False),
            _Norm2d(mid, norm=norm, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(mid, in_ch_logits, 1)
        self.apply(init_kaiming)

    def forward(self, image, logits_up):
        gray = self.to_gray(image)
        edge = self.hf(gray)
        x = torch.cat([logits_up, gray, edge], dim=1)
        x = self.fuse(x)
        delta = self.out(x)
        return logits_up + delta


# ---------------------------------------------------------
# Integer-aligned patch extract (general C)
# ---------------------------------------------------------
@torch.no_grad()
def extract_patches_from_centers_int_general(
    x_bchw: torch.Tensor,          # [B,C,H,W]
    centers_bk2: torch.Tensor,     # [B,K,2] in [-1,1]
    P: int,
) -> torch.Tensor:
    device = x_bchw.device
    B, C, H, W = x_bchw.shape
    K = centers_bk2.shape[1]
    pad = P // 2
    Hp = H + 2 * pad
    Wp = W + 2 * pad
    S = Hp * Wp

    centers01 = (centers_bk2 + 1.0) * 0.5
    cx = torch.round(centers01[..., 0] * (W - 1)).long().clamp(0, W - 1)
    cy = torch.round(centers01[..., 1] * (H - 1)).long().clamp(0, H - 1)

    x_pad = F.pad(x_bchw, (pad, pad, pad, pad), mode="constant", value=0.0)
    x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
    x_flat = x_pad.view(B, S, C)

    x0 = cx - (P // 2) + pad
    y0 = cy - (P // 2) + pad
    base = (y0 * Wp + x0).reshape(-1)

    dy = torch.arange(P, device=device, dtype=torch.long)
    dx = torch.arange(P, device=device, dtype=torch.long)
    yy, xx = _meshgrid_ij(dy, dx)
    off = (yy * Wp + xx).reshape(-1)
    idx2d = base[:, None] + off[None, :]

    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, K).reshape(-1)
    patches = x_flat[b_idx[:, None], idx2d, :]
    patches = patches.permute(0, 2, 1).contiguous().view(B * K, C, P, P)
    return patches


# ---------------------------------------------------------
# ROI center selection from coarse prob (pool @ stride)
# ---------------------------------------------------------
@torch.no_grad()
def select_roi_centers_from_coarse_prob(
    prob_full: torch.Tensor,  # [B,1,H,W] in [0,1]
    stride: int = 16,
    thr: float = 0.3,
    step: int = 1,
    max_rois: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = prob_full.device
    B, _, H, W = prob_full.shape
    stride = max(1, int(stride))
    step = max(1, int(step))
    max_rois = max(1, int(max_rois))

    prob_s = F.max_pool2d(prob_full, kernel_size=stride, stride=stride)
    B, _, Hs, Ws = prob_s.shape

    centers_out = torch.zeros((B, max_rois, 2), device=device, dtype=prob_full.dtype)
    valid_out = torch.zeros((B, max_rois), device=device, dtype=torch.bool)

    for b in range(B):
        p = prob_s[b, 0]
        ys, xs = torch.nonzero(p > thr, as_tuple=True)
        if ys.numel() == 0:
            flat_idx = torch.argmax(p.reshape(-1))
            ys = (flat_idx // Ws).view(1)
            xs = (flat_idx % Ws).view(1)

        vals = p[ys, xs]
        if vals.numel() > max_rois:
            topv, topi = torch.topk(vals, k=max_rois, largest=True, sorted=True)
            ys, xs, vals = ys[topi], xs[topi], topv

        if step > 1 and ys.numel() > 1:
            sel = torch.arange(0, ys.numel(), step, device=device, dtype=torch.long)
            ys, xs, vals = ys[sel], xs[sel], vals[sel]

        n = min(int(ys.numel()), max_rois)
        ys, xs = ys[:n], xs[:n]

        cx = xs.float() * stride + (stride / 2.0)
        cy = ys.float() * stride + (stride / 2.0)
        cx = cx.clamp(0, W - 1)
        cy = cy.clamp(0, H - 1)

        x_norm = (cx / (W - 1)).clamp(0, 1) * 2 - 1
        y_norm = (cy / (H - 1)).clamp(0, 1) * 2 - 1

        centers_out[b, :n, 0] = x_norm
        centers_out[b, :n, 1] = y_norm
        valid_out[b, :n] = True

    return centers_out, valid_out


# ---------------------------------------------------------
# Weight cache for paste
# ---------------------------------------------------------
_WEIGHT_CACHE = {}
_OFFSET_CACHE = {}

def _make_distance_weight(win: int, mode: str = "gaussian", sigma: float = None, device=None):
    key = (int(win), str(mode), float(sigma) if sigma is not None else None, str(device))
    if key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[key]

    yy = torch.arange(win, device=device).float()
    xx = torch.arange(win, device=device).float()
    yy, xx = _meshgrid_ij(yy, xx)

    cy = (win - 1) / 2.0
    cx = (win - 1) / 2.0
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = r / (r.max().clamp_min(1e-6))

    mode = mode.lower()
    if mode == "gaussian":
        if sigma is None:
            sigma = 0.35
        w = torch.exp(-(r_norm ** 2) / (2 * sigma * sigma))
    elif mode == "cosine":
        w = torch.cos((math.pi / 2.0) * r_norm).clamp_min(0.0) ** 2
    elif mode == "linear":
        w = (1.0 - r_norm).clamp_min(0.0)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    w = w.clamp_min(1e-6)
    _WEIGHT_CACHE[key] = w
    return w


def _get_patch_offsets(P: int, Wp: int, device):
    key = (int(P), int(Wp), str(device))
    if key in _OFFSET_CACHE:
        return _OFFSET_CACHE[key]
    dy = torch.arange(P, device=device, dtype=torch.long)
    dx = torch.arange(P, device=device, dtype=torch.long)
    dy, dx = _meshgrid_ij(dy, dx)
    off = (dy * Wp + dx).reshape(-1)
    _OFFSET_CACHE[key] = off
    return off


@torch.no_grad()
def paste_patches_weighted_fullres_fast(
    saliency_map_1hw: torch.Tensor,  # [B,1,H,W] logits (for thr)
    patch_logits: torch.Tensor,      # [B*K,1,P,P]
    centers: torch.Tensor,           # [B,K,2] in [-1,1]
    H: int, W: int,
    thr: float = 0.0,
    weight_mode: str = "gaussian",
    weight_sigma: float = 0.35,
    blend_space: str = "logit",      # "logit" or "prob"
    eps: float = 1e-6,
    valid_mask_bk: Optional[torch.Tensor] = None,   # [B,K] bool
):
    device = saliency_map_1hw.device
    B = int(saliency_map_1hw.shape[0])
    K = int(centers.shape[1])
    P = int(patch_logits.shape[-1])
    pad = P // 2

    Hp = H + 2 * pad
    Wp = W + 2 * pad
    S = Hp * Wp

    centers01 = (centers + 1.0) * 0.5
    cx = torch.round(centers01[:, :, 0] * (W - 1)).long().clamp(0, W - 1)
    cy = torch.round(centers01[:, :, 1] * (H - 1)).long().clamp(0, H - 1)

    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, K).reshape(-1)
    cx_flat = cx.reshape(-1)
    cy_flat = cy.reshape(-1)

    sval = torch.sigmoid(saliency_map_1hw[b_idx, 0, cy_flat, cx_flat])
    keep = (sval > thr)
    if valid_mask_bk is not None:
        keep = keep & valid_mask_bk.reshape(-1).to(torch.bool)

    if keep.sum().item() == 0:
        return torch.zeros((B, 1, H, W), device=device, dtype=patch_logits.dtype)

    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    b_keep = b_idx[keep_idx]
    cx_keep = cx_flat[keep_idx]
    cy_keep = cy_flat[keep_idx]

    patches = patch_logits.view(B * K, 1, P, P)[keep_idx]
    if blend_space == "prob":
        patches_v = torch.sigmoid(patches)
    else:
        patches_v = patches

    w_full = _make_distance_weight(P, mode=weight_mode, sigma=weight_sigma, device=device)
    w_vec = w_full.reshape(1, -1)

    x0 = cx_keep - (P // 2) + pad
    y0 = cy_keep - (P // 2) + pad
    base = y0 * Wp + x0

    off = _get_patch_offsets(P, Wp, device=device)
    idx2d = base.view(-1, 1) + off.view(1, -1)
    global_idx = (b_keep.view(-1, 1) * S + idx2d).reshape(-1)

    v = (patches_v[:, 0].reshape(-1, P * P) * w_vec).reshape(-1)
    wv = (torch.ones_like(patches_v[:, 0]).reshape(-1, P * P) * w_vec).reshape(-1)

    acc = torch.zeros((B * S,), device=device, dtype=patch_logits.dtype)
    wsum = torch.zeros((B * S,), device=device, dtype=patch_logits.dtype)

    acc.index_add_(0, global_idx, v)
    wsum.index_add_(0, global_idx, wv)

    out_pad = (acc / (wsum + eps)).view(B, Hp, Wp)
    out = out_pad[:, pad:pad + H, pad:pad + W].unsqueeze(1)

    if blend_space == "prob":
        out = out.clamp(1e-6, 1 - 1e-6)
        out = torch.log(out) - torch.log(1 - out)

    return out


# ---------------------------------------------------------
# Main model pieces
# ---------------------------------------------------------
class SaliencyAuxHead(nn.Module):
    def __init__(self, in_ch: int, mid: int = 128, norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        self.net = nn.Conv2d(in_ch, 1, 1, bias=True)
        self.apply(init_kaiming)
        nn.init.normal_(self.net.weight, mean=0.0, std=0.01)
        if self.net.bias is not None:
            nn.init.zeros_(self.net.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class InfMAE_CoarseToFine(nn.Module):
    """
    Two-stage model:
      - Stage1: full image (InfMAE backbone + UPerHead + optional BMSR)
      - Stage2: ROI patch refinement with UNet
    """
    def __init__(
        self,
        num_classes: int = 1,
        features: int = 256,
        crop_size_base: int = 33,
        up_size: int = 64,

        freeze_backbone: bool = True,
        pretrain_path: Optional[str] = None,
        target_input_size=512,
        # Adapter knobs (kept names, now used for PGRA rank ratio)
        use_adapter: bool = True,
        adapter_mlp_ratio_token: float = 0.25,
        adapter_mlp_ratio_2d: float = 0.25,
        adapter_gn_groups: int = 16,
        adapter_k_stat: int = 3,
        adapter_detach_gate: bool = True,

        # ROI
        thr_s16: float = 0.3,
        stride_roi: int = 16,
        step_s16: int = 1,
        max_rois: int = 256,
        patch_chunk: int = 64,

        # paste / gating
        saliency_thr: float = 0.0,

        # BMSR
        use_bmsr: bool = True,
        bmsr_norm: str = "gn",
        bmsr_gn_groups: int = 16,

        # Stage2 patch refinement
        use_boundary_refine: bool = True,
        patch_use_imagenet: bool = True,

        # loss
        pos_weight: float = 9.0,
        lam_global_up: float = 1.0,
        lam_global_ref: float = 1.0,
        lam_patch: float = 2.0,
        lam_patch_pred: float = 1.0,        # raw
        lam_patch_ref: float = 1.0,         # ✅ ref（默认给1，避免你忘了开）

        # misc
        detach_base_patch: bool = True,

        # Saliency auxiliary supervision
        use_sal_aux: bool = False,
        sal_mid: int = 128,
        lam_sal_f2: float = 0.5,
        lam_sal_f4: float = 0.5,
        pos_weight_sal: Optional[float] = None,

        # Multi-task loss wrapper
        use_mt_loss: bool = True,
    ):
        super().__init__()
        '''
        如果 Fa 还是高：
        lambda_fp_global = 0.5
        lambda_fp_patch = 1.0
        
        如果 Pd 掉太多：
        lambda_fp_global = 0.2
        lambda_fp_patch = 0.5
        '''
        self.lambda_fp_global = 0.3
        self.lambda_fp_patch = 0.8

        self.crop_size_base = int(crop_size_base)
        self.up_size = int(up_size)

        self.thr_s16 = float(thr_s16)
        self.stride_roi = int(stride_roi)
        self.step_s16 = int(step_s16)
        self.max_rois = int(max_rois)
        self.patch_chunk = int(patch_chunk)

        self.saliency_thr = float(saliency_thr)

        self.use_bmsr = bool(use_bmsr)
        self.use_boundary_refine = bool(use_boundary_refine)

        self.pos_weight = float(pos_weight)
        self.lam_global_up = float(lam_global_up)
        self.lam_global_ref = float(lam_global_ref)
        self.lam_patch = float(lam_patch)
        self.lam_patch_pred = float(lam_patch_pred)
        self.lam_patch_ref = float(lam_patch_ref)

        self.detach_base_patch = bool(detach_base_patch)

        if isinstance(target_input_size, (tuple, list)):
            if len(target_input_size) != 3:
                raise ValueError("target_input_size must be an integer or a three-level size sequence")
            encoder_sizes = [int(size) for size in target_input_size]
        else:
            input_size = int(target_input_size)
            encoder_sizes = [input_size, input_size // 4, input_size // 8]

        # ✅ multi-task wrappers
        self.use_mt_loss = bool(use_mt_loss)
        if self.use_mt_loss:
            self.mt_stage1 = MultiTaskLossWrapper(task_num=2)  # [global_up, global_ref]
            # ✅ Stage2: boundary_refine -> 2 tasks，否则 1 task（避免多出来的log-term常数）
            self.mt_stage2 = MultiTaskLossWrapper(task_num=2 if self.use_boundary_refine else 1)


        self.encoder = MaskedAutoencoderInfMAE_PGRAAdapter(
            img_size=encoder_sizes,
            patch_size=[4, 2, 2],
            in_chans=3,
            embed_dim=[256, 384, 768],
            depth=[2, 2, 11],
            num_heads=12,
            decoder_embed_dim=512,
            decoder_depth=2,
            decoder_num_heads=16,
            mlp_ratio=[4, 4, 4],
            norm_layer=nn.LayerNorm,
            norm_pix_loss=False,

            adapter_mlp_ratio_token=adapter_mlp_ratio_token,
            adapter_mlp_ratio_2d=adapter_mlp_ratio_2d,
            adapter_gn_groups=adapter_gn_groups,
            adapter_k_stat=adapter_k_stat,
            adapter_detach_gate=adapter_detach_gate,
        )

        # ---- Load pretrain FIRST ----
        if pretrain_path is not None:
            load_infmae_pretrain(self.encoder, pretrain_path, target_input_size=target_input_size, strict=False)

        # ---- Inject adapters ----
        if use_adapter:
            self.encoder.inject_pgra_adapters()

        # ---- Freeze backbone but keep adapters trainable ----
        if freeze_backbone:
            if use_adapter:
                freeze_infmae_except_pgra_adapters(self.encoder)
            else:
                for p in self.encoder.parameters():
                    p.requires_grad = False

        self.backbone = InfMAEBackbone(self.encoder)

        # ---- saliency aux supervision ----
        self.use_sal_aux = bool(use_sal_aux)
        self.lam_sal_f2 = float(lam_sal_f2)
        self.lam_sal_f4 = float(lam_sal_f4)
        self.pos_weight_sal = float(pos_weight_sal) if pos_weight_sal is not None else None

        if self.use_sal_aux:
            self.sal_head_f2 = SaliencyAuxHead(in_ch=384, mid=sal_mid, norm="gn", gn_groups=16)
            self.sal_head_f4 = SaliencyAuxHead(in_ch=768, mid=sal_mid, norm="gn", gn_groups=16)

        # ---- Global head: UPerNet ----
        self.global_head = UPerHead(
            in_channels=[256, 384, 768, 768],
            channels=features,
            num_classes=num_classes,
            ppm_pool_scales=(1, 2, 3, 6),
            align_corners=True,
        )

        # ---- BMSR ----
        self.refiner = BMSRRefiner(
            in_ch_image=3, in_ch_logits=num_classes, mid=64, k=3,
            norm=bmsr_norm, gn_groups=bmsr_gn_groups
        )

        self.patch_sobel = SobelHF()
        self.patch_unet = DualBranchLiteUNetV2(
            use_boundary_refine=self.use_boundary_refine,
            use_imagenet=patch_use_imagenet,
        )

    def _select_fg(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            return logits[:, :1]
        return logits[:, 1:].max(dim=1, keepdim=True).values

    def forward_stage1(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, _, H, W = x.shape
        feats = self.backbone(x)
        f1, f2, f3, f4 = feats

        sal_f2 = None
        sal_f4 = None
        if getattr(self, "use_sal_aux", False):
            sal_f2 = self.sal_head_f2(f2)
            sal_f4 = self.sal_head_f4(f4)

        global_low = self.global_head([f1, f2, f3, f4])
        global_up = F.interpolate(global_low, size=(H, W), mode="bilinear", align_corners=True)

        if self.use_bmsr:
            global_ref = self.refiner(x, global_up)
        else:
            global_ref = global_up

        global_ref_fg = self._select_fg(global_ref)
        aux = {
            "global_up": global_up,
            "global_ref": global_ref,
            "global_ref_fg": global_ref_fg,
            "sal_f2": sal_f2,
            "sal_f4": sal_f4,
        }
        return global_ref_fg, aux

    def forward_stage2(self, x: torch.Tensor, global_ref_fg: torch.Tensor):
        B, _, H, W = x.shape
        Pz = int(self.crop_size_base)
        U = int(self.up_size)

        prob_full = torch.sigmoid(global_ref_fg)

        centers_full, valid_bk = select_roi_centers_from_coarse_prob(
            prob_full=prob_full,
            stride=self.stride_roi,
            thr=self.thr_s16,
            step=self.step_s16,
            max_rois=self.max_rois,
        )
        K = int(centers_full.shape[1])

        base_patch_all = extract_patches_from_centers_int_general(global_ref_fg, centers_full, Pz)
        img_patch_all  = extract_patches_from_centers_int_general(x, centers_full, Pz)

        if self.detach_base_patch:
            base_patch_all = base_patch_all.detach()

        valid_flat = valid_bk.reshape(-1)

        final_patch_all = base_patch_all.clone()
        pred_patch_all  = torch.zeros_like(base_patch_all)
        delta_patch_all = torch.zeros_like(base_patch_all)

        patch_pred_up_valid = None  # raw
        patch_ref_up_valid  = None  # refined
        valid_indices = None

        if valid_flat.any():
            valid_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)

            base_patch = base_patch_all[valid_flat]
            img_patch  = img_patch_all[valid_flat]

            if U == Pz:
                img_up = img_patch
                base_up = base_patch
            else:
                img_up  = F.interpolate(img_patch,  size=(U, U), mode="bilinear", align_corners=True)
                base_up = F.interpolate(base_patch, size=(U, U), mode="bilinear", align_corners=True)  # logits

            base_prob = torch.sigmoid(base_up)
            base_edge = self.patch_sobel(base_prob)
            base_edge = base_edge / (base_edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))

            inp = torch.cat([img_up, base_prob, base_edge], dim=1)

            raw_list, ref_list = [], []
            chunk = max(1, int(self.patch_chunk))
            for s in range(0, inp.shape[0], chunk):
                out_s = self.patch_unet(inp[s:s + chunk])

                if isinstance(out_s, dict):
                    raw_s = out_s.get("logits_raw", None)
                    ref_s = out_s.get("logits_ref", None)
                    if raw_s is None:
                        raw_s = out_s.get("logits", None)
                    if ref_s is None:
                        ref_s = raw_s
                    if raw_s is None:
                        raise KeyError("PatchUNet dict output must contain logits_raw/logits_ref (or logits as fallback).")
                else:
                    raw_s = out_s
                    ref_s = out_s

                raw_list.append(raw_s)
                ref_list.append(ref_s)

            patch_pred_up_valid = torch.cat(raw_list, dim=0)  # raw [Nv,1,U,U]
            patch_ref_up_valid  = torch.cat(ref_list, dim=0)  # ref [Nv,1,U,U]


            if U == Pz:
                pred_patch = patch_pred_up_valid
                final_patch = patch_ref_up_valid
            else:
                pred_patch = F.interpolate(patch_pred_up_valid, size=(Pz, Pz), mode="bilinear", align_corners=True)
                final_patch = F.interpolate(patch_ref_up_valid, size=(Pz, Pz), mode="bilinear", align_corners=True)

            delta_patch = final_patch - pred_patch

            pred_patch_all[valid_flat]  = pred_patch
            final_patch_all[valid_flat] = final_patch
            delta_patch_all[valid_flat] = delta_patch

        ones_patch = torch.ones_like(final_patch_all)
        support_logits = paste_patches_weighted_fullres_fast(
            saliency_map_1hw=global_ref_fg,
            patch_logits=ones_patch,
            centers=centers_full,
            H=H, W=W,
            thr=self.saliency_thr,
            weight_mode="gaussian",
            weight_sigma=0.35,
            blend_space="logit",
            valid_mask_bk=valid_bk,
        )
        support = (support_logits.abs() > 1e-12).float()

        patched_logits = paste_patches_weighted_fullres_fast(
            saliency_map_1hw=global_ref_fg,
            patch_logits=final_patch_all,
            centers=centers_full,
            H=H, W=W,
            thr=self.saliency_thr,
            weight_mode="gaussian",
            weight_sigma=0.35,
            blend_space="logit",
            valid_mask_bk=valid_bk,
        )
        refined_full = global_ref_fg * (1.0 - support) + patched_logits * support

        aux = {
            "grid_centers": centers_full,
            "valid_bk": valid_bk,
            "valid_indices": valid_indices,
            "K": K,
            "patch_size": Pz,
            "up_size": U,

            "patch_base_zoom": base_patch_all,
            "patch_pred_zoom": pred_patch_all,
            "patch_delta_zoom": delta_patch_all,
            "patch_final_zoom": final_patch_all,
            "support": support,

            # ✅ stage2 supervision targets:
            "patch_pred_up_valid": patch_pred_up_valid,  # raw
            "patch_ref_up_valid": patch_ref_up_valid,    # refined (may equal raw)
        }
        return refined_full, aux

    def forward(self, x: torch.Tensor, gt_mask: torch.Tensor = None, coarse_only: bool = False):
        global_ref_fg, aux1 = self.forward_stage1(x)

        if coarse_only:
            return {"logits": global_ref_fg, "aux": aux1}

        refined_full, aux2 = self.forward_stage2(x, global_ref_fg)
        aux = {}
        aux.update(aux1)
        aux.update(aux2)
        return {"logits": refined_full, "aux": aux}

    def compute_loss(self, out: Dict, gt_mask: torch.Tensor, stage: int = 1) -> Dict:
        aux = out["aux"]
        device = gt_mask.device
        gt_full = _to_mask_4d(gt_mask, device=device)

        # -------------------------
        # Stage 1
        # -------------------------
        if int(stage) == 1:
            global_up = aux["global_up"]          # logits [B,1,H,W]
            global_ref_fg = aux["global_ref_fg"]  # logits [B,1,H,W]
            global_up_fg = self._select_fg(global_up)

            loss_global_up = seg_bce_dice_loss(global_up_fg, gt_full, pos_weight=self.pos_weight)
            loss_global_ref = seg_bce_dice_loss(global_ref_fg, gt_full, pos_weight=self.pos_weight)

            if self.use_mt_loss:
                total = self.mt_stage1([loss_global_up, loss_global_ref])
            else:
                total = self.lam_global_up * loss_global_up + self.lam_global_ref * loss_global_ref

            # saliency aux stays additive (not inside wrapper)
            loss_sal_f2 = torch.zeros((), device=device)
            loss_sal_f4 = torch.zeros((), device=device)

            if getattr(self, "use_sal_aux", False):
                pw_sal = self.pos_weight_sal if (self.pos_weight_sal is not None) else self.pos_weight * 1.5

                sal_f2 = aux.get("sal_f2", None)
                if sal_f2 is not None:
                    gt_f2 = maxpool_to_match(gt_full, sal_f2)
                    loss_sal_f2 = seg_bce_dice_loss(sal_f2, gt_f2, pos_weight=pw_sal)
                    total = total + self.lam_sal_f2 * loss_sal_f2

                sal_f4 = aux.get("sal_f4", None)
                if sal_f4 is not None:
                    gt_f4 = maxpool_to_match(gt_full, sal_f4)
                    loss_sal_f4 = seg_bce_dice_loss(sal_f4, gt_f4, pos_weight=pw_sal)
                    total = total + self.lam_sal_f4 * loss_sal_f4

            return {
                "loss": total,
                "loss_global_up": loss_global_up,
                "loss_global_ref": loss_global_ref,
                "loss_sal_f2": loss_sal_f2,
                "loss_sal_f4": loss_sal_f4,
            }

        # -------------------------
        # Stage 2
        # -------------------------
        centers = aux.get("grid_centers", None)
        valid_bk = aux.get("valid_bk", None)
        Pz = int(aux.get("patch_size", self.crop_size_base))
        U = int(aux.get("up_size", self.up_size))

        pred_patch_up = aux.get("patch_pred_up_valid", None)  # raw [Nv,1,U,U]
        ref_patch_up  = aux.get("patch_ref_up_valid", None)   # ref [Nv,1,U,U]
        valid_idx = aux.get("valid_indices", None)

        # ✅ Important: when no valid patches, return 0 (do NOT apply mt wrapper, otherwise adds constant log-term)
        if (centers is None) or (valid_bk is None) or (pred_patch_up is None) or (valid_idx is None) or (valid_idx.numel() == 0):
            zero = torch.zeros((), device=device, requires_grad=True)
            return {
                "loss": zero,
                "loss_patch_pred": zero.detach(),
                "loss_patch_ref": zero.detach(),
            }

        # 兜底：如果没ref分支（或没开），当作ref=raw
        if ref_patch_up is None:
            ref_patch_up = pred_patch_up

        with torch.no_grad():
            gt_patch_all = extract_patches_from_centers_int_general(gt_full, centers, Pz)  # [BK,1,Pz,Pz]
            gt_patch_valid = gt_patch_all.view(-1, 1, Pz, Pz)[valid_idx]                   # [Nv,1,Pz,Pz]
            gt_up = F.interpolate(gt_patch_valid.float(), size=(U, U), mode="nearest")     # [Nv,1,U,U]

        loss_patch_pred = seg_bce_dice_loss(pred_patch_up, gt_up, pos_weight=self.pos_weight)

        if self.use_boundary_refine:
            loss_patch_ref = seg_bce_dice_loss(ref_patch_up, gt_up, pos_weight=self.pos_weight)

            if self.use_mt_loss:
                total = self.mt_stage2([loss_patch_pred, loss_patch_ref])
            else:
                total = self.lam_patch * (
                    self.lam_patch_pred * loss_patch_pred + self.lam_patch_ref * loss_patch_ref
                )
        else:
            loss_patch_ref = torch.zeros((), device=device)
            if self.use_mt_loss:
                total = self.mt_stage2([loss_patch_pred])
            else:
                total = self.lam_patch * (self.lam_patch_pred * loss_patch_pred)

        return {
            "loss": total,
            "loss_patch_pred": loss_patch_pred,
            "loss_patch_ref": loss_patch_ref,
        }

    def compute_loss_new(self, out: Dict, gt_mask: torch.Tensor, stage: int = 1) -> Dict:
        aux = out["aux"]
        device = gt_mask.device
        gt_full = _to_mask_4d(gt_mask, device=device)

        # 建议：global 分支弱一点，patch 分支强一点
        lambda_fp_global = getattr(self, "lambda_fp_global", 0.3)
        lambda_fp_patch = getattr(self, "lambda_fp_patch", 0.8)

        # -------------------------
        # Stage 1
        # -------------------------
        if int(stage) == 1:
            global_up = aux["global_up"]  # logits [B,1,H,W]
            global_ref_fg = aux["global_ref_fg"]  # logits [B,1,H,W]
            global_up_fg = self._select_fg(global_up)

            loss_global_up = seg_bce_dice_fp_loss(
                global_up_fg,
                gt_full,
                pos_weight=self.pos_weight,
                lambda_fp=lambda_fp_global,
            )

            loss_global_ref = seg_bce_dice_fp_loss(
                global_ref_fg,
                gt_full,
                pos_weight=self.pos_weight,
                lambda_fp=lambda_fp_global,
            )

            if self.use_mt_loss:
                total = self.mt_stage1([loss_global_up, loss_global_ref])
            else:
                total = self.lam_global_up * loss_global_up + self.lam_global_ref * loss_global_ref

            loss_sal_f2 = torch.zeros((), device=device)
            loss_sal_f4 = torch.zeros((), device=device)

            if getattr(self, "use_sal_aux", False):
                pw_sal = self.pos_weight_sal if (self.pos_weight_sal is not None) else self.pos_weight * 1.5

                sal_f2 = aux.get("sal_f2", None)
                if sal_f2 is not None:
                    gt_f2 = maxpool_to_match(gt_full, sal_f2)
                    loss_sal_f2 = seg_bce_dice_fp_loss(
                        sal_f2,
                        gt_f2,
                        pos_weight=pw_sal,
                        lambda_fp=lambda_fp_global,
                    )
                    total = total + self.lam_sal_f2 * loss_sal_f2

                sal_f4 = aux.get("sal_f4", None)
                if sal_f4 is not None:
                    gt_f4 = maxpool_to_match(gt_full, sal_f4)
                    loss_sal_f4 = seg_bce_dice_fp_loss(
                        sal_f4,
                        gt_f4,
                        pos_weight=pw_sal,
                        lambda_fp=lambda_fp_global,
                    )
                    total = total + self.lam_sal_f4 * loss_sal_f4

            return {
                "loss": total,
                "loss_global_up": loss_global_up,
                "loss_global_ref": loss_global_ref,
                "loss_sal_f2": loss_sal_f2,
                "loss_sal_f4": loss_sal_f4,
            }

        # -------------------------
        # Stage 2
        # -------------------------
        centers = aux.get("grid_centers", None)
        valid_bk = aux.get("valid_bk", None)
        Pz = int(aux.get("patch_size", self.crop_size_base))
        U = int(aux.get("up_size", self.up_size))

        pred_patch_up = aux.get("patch_pred_up_valid", None)
        ref_patch_up = aux.get("patch_ref_up_valid", None)
        valid_idx = aux.get("valid_indices", None)

        if (
                (centers is None)
                or (valid_bk is None)
                or (pred_patch_up is None)
                or (valid_idx is None)
                or (valid_idx.numel() == 0)
        ):
            zero = torch.zeros((), device=device, requires_grad=True)
            return {
                "loss": zero,
                "loss_patch_pred": zero.detach(),
                "loss_patch_ref": zero.detach(),
            }

        if ref_patch_up is None:
            ref_patch_up = pred_patch_up

        with torch.no_grad():
            gt_patch_all = extract_patches_from_centers_int_general(gt_full, centers, Pz)
            gt_patch_valid = gt_patch_all.view(-1, 1, Pz, Pz)[valid_idx]
            gt_up = F.interpolate(gt_patch_valid.float(), size=(U, U), mode="nearest")

        loss_patch_pred = seg_bce_dice_fp_loss(
            pred_patch_up,
            gt_up,
            pos_weight=self.pos_weight,
            lambda_fp=lambda_fp_patch,
        )

        if self.use_boundary_refine:
            loss_patch_ref = seg_bce_dice_fp_loss(
                ref_patch_up,
                gt_up,
                pos_weight=self.pos_weight,
                lambda_fp=lambda_fp_patch,
            )

            if self.use_mt_loss:
                total = self.mt_stage2([loss_patch_pred, loss_patch_ref])
            else:
                total = self.lam_patch * (
                        self.lam_patch_pred * loss_patch_pred
                        + self.lam_patch_ref * loss_patch_ref
                )
        else:
            loss_patch_ref = torch.zeros((), device=device)

            if self.use_mt_loss:
                total = self.mt_stage2([loss_patch_pred])
            else:
                total = self.lam_patch * (self.lam_patch_pred * loss_patch_pred)

        return {
            "loss": total,
            "loss_patch_pred": loss_patch_pred,
            "loss_patch_ref": loss_patch_ref,
        }

def maxpool_to_match(gt_full_1chw: torch.Tensor, ref_1chw: torch.Tensor) -> torch.Tensor:
    H, W = gt_full_1chw.shape[-2:]
    h, w = ref_1chw.shape[-2:]

    if (H % h == 0) and (W % w == 0):
        kh, kw = H // h, W // w
        return F.max_pool2d(gt_full_1chw, kernel_size=(kh, kw), stride=(kh, kw))

    else:
        return F.adaptive_max_pool2d(gt_full_1chw, output_size=(h, w))


# Backward-compatible name used by the original training and inference scripts.
PGSRNet = InfMAE_CoarseToFine
