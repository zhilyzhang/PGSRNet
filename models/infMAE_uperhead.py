
from typing import List, Tuple, Optional
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models_infmae_skip4 import MaskedAutoencoderInfMAE


def init_kaiming(module: torch.nn.Module):
    """
    对卷积/线性做 Kaiming 正态初始化，BN 权重=1、偏置=0。
    仅对未加载预训练的“新层”使用，避免覆盖主干权重。
    """
    if isinstance(module, nn.Conv2d):
        # 对应 ReLU 的 kaiming 正态
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.apply(init_kaiming)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



class UPerHead(nn.Module):
    """
    UPerNet 风格分割头：
      - 输入: 4 个尺度特征 [C1(1/4), C2(1/8), C3(1/16), C4(1/16)]
      - PPM 在最深层 (C4) 上做金字塔池化
      - 自顶向下 FPN 融合
      - 将各尺度统一到最高分辨率(1/4)后 concat，最后输出 num_classes
    """
    def __init__(
        self,
        in_channels: List[int],   # 例如 [256, 384, 768, 768]
        channels: int = 256,
        num_classes: int = 6,
        ppm_pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
        align_corners: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.ppm_pool_scales = ppm_pool_scales
        self.align_corners = align_corners

        # 1) lateral convs（把每级通道都压到 channels）
        self.lateral_convs = nn.ModuleList([
            ConvBNReLU(c, channels, k=1, s=1, p=0) for c in in_channels
        ])

        # 2) PPM on top (最后一层)
        ppm_in = channels
        self.ppm_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvBNReLU(ppm_in, channels, k=1, s=1, p=0)
            )
            for scale in ppm_pool_scales
        ])
        self.ppm_bottleneck = ConvBNReLU(ppm_in*(len(ppm_pool_scales)+1), channels, k=3, s=1, p=1)

        # 3) FPN 的顶到底 3 个合并卷积（不改变 channels）
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(channels, channels, k=3, s=1, p=1) for _ in range(len(in_channels)-1)
        ])

        # 4) FPN 后的各层再来一个 3x3（保持 channels）
        self.fpn_out = nn.ModuleList([
            ConvBNReLU(channels, channels, k=3, s=1, p=1) for _ in range(len(in_channels)-1)
        ])

        # 5) 最终融合各尺度到最高分辨率(1/4)后 concat
        self.fuse = ConvBNReLU(channels * len(in_channels), channels, k=3, s=1, p=1)
        self.cls  = nn.Conv2d(channels, num_classes, kernel_size=1)

        self.apply(init_kaiming)
        nn.init.normal_(self.cls.weight, mean=0.0, std=0.01)
        if self.cls.bias is not None:
            nn.init.zeros_(self.cls.bias)


    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        feats: [C1(1/4), C2(1/8), C3(1/16), C4(1/16)]
               每个是 [B, C, H, W]
        """
        assert len(feats) == 4
        c1, c2, c3, c4 = feats  # 尺度从高到低

        # —— lateral 对齐通道 —— #
        lat = [l(f) for l, f in zip(self.lateral_convs, [c1, c2, c3, c4])]
        l1, l2, l3, l4 = lat

        # —— PPM 在最深层 l4 —— #
        ppm_outs = [l4]
        for ppm in self.ppm_modules:
            ppm_outs.append(F.interpolate(ppm(l4), size=l4.shape[-2:], mode='bilinear', align_corners=self.align_corners))
        l4_ppm = self.ppm_bottleneck(torch.cat(ppm_outs, dim=1))  # 仍然是 1/16

        # —— Top-down FPN: l4→l3→l2→l1 —— #
        f3 = l3 + F.interpolate(l4_ppm, size=l3.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        f3 = self.fpn_convs[0](f3)

        f2 = l2 + F.interpolate(f3, size=l2.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        f2 = self.fpn_convs[1](f2)

        f1 = l1 + F.interpolate(f2, size=l1.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        f1 = self.fpn_convs[2](f1)

        # —— 再各自 refine 一下（可选，和 mmseg 接近）—— #
        f3 = self.fpn_out[0](f3)
        f2 = self.fpn_out[1](f2)
        f1 = self.fpn_out[2](f1)

        # —— 统一到最高分辨率(1/4)后 concat —— #
        h1w1 = f1.shape[-2:]
        up_f2 = F.interpolate(f2, size=h1w1, mode='bilinear', align_corners=self.align_corners)
        up_f3 = F.interpolate(f3, size=h1w1, mode='bilinear', align_corners=self.align_corners)
        up_f4 = F.interpolate(l4_ppm, size=h1w1, mode='bilinear', align_corners=self.align_corners)

        x = torch.cat([f1, up_f2, up_f3, up_f4], dim=1)  # B, 4*channels, H/4, W/4
        x = self.fuse(x)
        logits = self.cls(x)  # B, num_classes, H/4, W/4
        return logits

# ------------------------------
# InfMAEBackbone: 让主干以“特征提取模式”工作（无随机掩码）
# ------------------------------
class InfMAEBackbone(nn.Module):
    """
    使用 MaskedAutoencoderInfMAE 作为主干，
    并提供 forward_features(x) 输出 4 级 token 特征（均为 [B, L, C]，L=14*14，C=embed_dim[-1]）。
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        # 一些便捷引用
        self.pe1 = backbone.patch_embed1
        self.pe2 = backbone.patch_embed2
        self.pe3 = backbone.patch_embed3
        self.lin4 = backbone.patch_embed4
        self.blocks1 = backbone.blocks1
        self.blocks2 = backbone.blocks2
        self.blocks3 = backbone.blocks3
        self.stage1_decode = backbone.stage1_output_decode  # Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_decode = backbone.stage2_output_decode  # Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)
        self.pos_embed = backbone.pos_embed  # [1, L, C]
        self.norm = backbone.norm

        # 计算 token 网格大小（例如 14×14）
        self.num_patches = self.pe3.num_patches
        self.patch_h = self.patch_w = int(self.num_patches ** 0.5)
        assert self.patch_h * self.patch_w == self.num_patches

    @torch.no_grad()
    def _make_all_ones_mask(self, B: int, H: int, W: int) -> torch.Tensor:
        """
        生成 (1 - mask) 的形态：1 表示可见（keep），shape = [B, 1, H, W]
        """
        return torch.ones(B, 1, H, W, device=self.pos_embed.device, dtype=self.pos_embed.dtype)

    def forward_features(self, x: torch.Tensor):
        """
        返回 4 个尺度特征图：
          C1: 1/4  , 通道 = embed_dim[0]  (来自 x1)
          C2: 1/8  , 通道 = embed_dim[1]  (来自 x2)
          C3: 1/16 , 通道 = embed_dim[2]  (来自 x3 原始卷积输出)
          C4: 1/16 , 通道 = embed_dim[2]  (来自 tokens_post reshape 回 2D)
        """
        B, _, H, W = x.shape

        # stage1
        x1 = self.pe1(x)  # [B, C1, H/4, W/4]
        keep_mask1 = self._make_all_ones_mask(B, x1.shape[-2], x1.shape[-1])
        for blk in self.blocks1:
            x1 = blk(x1, keep_mask1)  # -> C1, stride 4

        # stage2
        x2 = self.pe2(x1)  # [B, C2, H/8, W/8]
        keep_mask2 = self._make_all_ones_mask(B, x2.shape[-2], x2.shape[-1])
        for blk in self.blocks2:
            x2 = blk(x2, keep_mask2)  # -> C2, stride 8

        # stage3
        x3 = self.pe3(x2)  # [B, C3, H/16, W/16]
        # ViT blocks3 在 token 维度上执行；我们构造 tokens -> blocks3 -> norm -> 再 reshape 回 2D
        tokens = x3.flatten(2).permute(0, 2, 1)  # [B, L, C3]
        tokens = self.lin4(tokens)
        tokens_pre = tokens + self.pos_embed[:, :tokens.shape[1], :]
        y = tokens_pre
        for blk in self.blocks3:
            y = blk(y)
        tokens_post = self.norm(y)  # [B, L, C3]
        # reshape 回 2D：H/16, W/16
        ph = x3.shape[-2]
        pw = x3.shape[-1]
        x4 = tokens_post.permute(0, 2, 1).reshape(B, x3.shape[1], ph, pw)  # C3, stride16

        # 返回 4 级 feature maps
        return [x1, x2, x3, x4]

    def forward_features_new(self, x: torch.Tensor):
        """
        返回 4 个尺度特征图：
          x1: 1/4  (来自 blocks1 输出)
          x2: 1/8  (来自 blocks2 输出)
          x3: 1/16 (来自 blocks3 前半段输出，reshape 回2D)
          x4: 1/16 (来自 blocks3 最终输出，reshape 回2D)
        """
        B, _, H, W = x.shape

        # stage1
        x1 = self.pe1(x)  # [B, C1, H/4, W/4]
        keep_mask1 = self._make_all_ones_mask(B, x1.shape[-2], x1.shape[-1])
        for blk in self.blocks1:
            x1 = blk(x1, keep_mask1)

        # stage2
        x2 = self.pe2(x1)  # [B, C2, H/8, W/8]
        keep_mask2 = self._make_all_ones_mask(B, x2.shape[-2], x2.shape[-1])
        for blk in self.blocks2:
            x2 = blk(x2, keep_mask2)

        # stage3 conv -> tokens
        x3_conv = self.pe3(x2)  # [B, C3, H/16, W/16]
        ph, pw = x3_conv.shape[-2], x3_conv.shape[-1]
        C3 = x3_conv.shape[1]

        tokens = x3_conv.flatten(2).permute(0, 2, 1)  # [B, L, C3]
        tokens = self.lin4(tokens)
        tokens_pre = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # blocks3：跑一遍并在“前半段结束”处截取中间特征
        n3 = len(self.blocks3)
        mid = n3 // 2  # e.g. 11 -> 5
        y = tokens_pre
        y_mid = tokens_pre  # mid==0 时，中间特征就取输入（即 0 个 block 后）

        for i, blk in enumerate(self.blocks3):
            y = blk(y)
            if i == mid - 1:
                y_mid = y

        # x3：blocks3 前半段输出（默认做同一个norm，但不影响后续y）
        tokens_mid = self.norm(y_mid)
        x3 = tokens_mid.permute(0, 2, 1).reshape(B, C3, ph, pw)

        # x4：blocks3 最终输出
        tokens_post = self.norm(y)
        x4 = tokens_post.permute(0, 2, 1).reshape(B, C3, ph, pw)

        return [x1, x2, x3, x4]

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], int, int]:
        return self.forward_features(x)


# ------------------------------
# 权重加载与位置编码插值
# ------------------------------
def interpolate_pos_embed(pe: torch.Tensor, target_hw: int) -> torch.Tensor:
    """
    pe: [1, L, C]
    target_hw: 目标的 H(=W)，例如 14
    """
    L, C = pe.shape[1], pe.shape[2]
    src_hw = int(L ** 0.5)
    if src_hw == target_hw:
        return pe
    # [1, C, H, W]
    pe_2d = pe[0].transpose(0, 1).reshape(1, C, src_hw, src_hw)
    pe_2d = F.interpolate(pe_2d, size=(target_hw, target_hw), mode='bicubic', align_corners=False)
    pe_new = pe_2d.reshape(1, C, target_hw * target_hw).transpose(1, 2)  # [1, L', C]
    return pe_new


def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)

@torch.no_grad()
def _set_patch_embed_imgsize(pe, img_size_hw):
    """安全更新 PatchEmbed/PatchEmbed_F 的 img_size / grid_size / num_patches"""
    if pe is None:
        return
    H, W = _to_2tuple(img_size_hw)
    if hasattr(pe, "img_size"):
        pe.img_size = (H, W)
    # 推断 patch_size
    ps = None
    if hasattr(pe, "patch_size"):
        ps = pe.patch_size
    elif hasattr(pe, "proj") and isinstance(pe.proj, nn.Conv2d):
        ps = pe.proj.kernel_size
    if ps is None:
        # 没有 patch_size（比如 PatchEmbed_F），也没关系，它一般只用 img_size 做断言
        return
    ph, pw = _to_2tuple(ps)
    gh, gw = H // ph, W // pw
    if hasattr(pe, "grid_size"):
        pe.grid_size = (gh, gw)
    if hasattr(pe, "num_patches"):
        pe.num_patches = gh * gw

def _interp_pos_embed(pe_ckpt, target_hw):
    """
    pe_ckpt: [1, Lsrc, C]
    target_hw: 目标 token 网格边长（例如 512 输入、累计步长16 → 32）
    返回 [1, Ldst, C]
    """
    if pe_ckpt is None:
        return None
    pe = pe_ckpt
    if isinstance(pe, torch.Tensor):
        pass
    else:
        pe = torch.tensor(pe)
    assert pe.ndim == 3 and pe.shape[0] == 1
    Lsrc, C = pe.shape[1], pe.shape[2]
    hsrc = int(math.sqrt(Lsrc))
    assert hsrc * hsrc == Lsrc, "pos_embed length must be a square"
    if hsrc == target_hw:
        return pe
    # [1,C,H,W] → 插值 → [1,L,C]
    pe_2d = pe[0].transpose(0, 1).reshape(1, C, hsrc, hsrc)
    pe_2d = F.interpolate(pe_2d, size=(target_hw, target_hw), mode='bicubic', align_corners=False)
    pe_new = pe_2d.reshape(1, C, target_hw * target_hw).transpose(1, 2).contiguous()
    return pe_new


def load_infmae_pretrain(model, ckpt_path, target_input_size=224, strict=False):
    """
    根据 target_input_size（例如 512），把 224 的预训练权重位置编码插值到新网格；
    同步更新 model 内各 PatchEmbed 的 img_size / num_patches；
    同时处理 decoder_pos_embed。
    """
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'model' in ckpt and isinstance(ckpt['model'], dict):
            state = ckpt['model']
        elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state = ckpt['state_dict']
        else:
            state = ckpt
    else:
        state = ckpt

    # ===== 1) 计算目标各 stage 的特征大小（累计步长 = 4*2*2 = 16） =====
    H = int(target_input_size)
    assert H % 16 == 0, "target_input_size must be divisible by 16"
    s1 = H // 4   # stage1 feature map 边长（224→56；512→128）
    s2 = H // 8   # stage2 feature map 边长（224→28；512→64）
    s3 = H // 16  # 最终 token 网格边长（224→14；512→32）

    # ===== 2) 更新主干中所有 PatchEmbed 的 img_size / grid_size / num_patches =====
    _set_patch_embed_imgsize(getattr(model, "patch_embed1", None), (H, H))
    _set_patch_embed_imgsize(getattr(model, "patch_embed2", None), (s1, s1))
    _set_patch_embed_imgsize(getattr(model, "patch_embed3", None), (s2, s2))
    _set_patch_embed_imgsize(getattr(model, "patch_embed",  None), (H, H))  # PatchEmbed_F（若 forward 会用到）

    # ===== 3) 创建与目标网格一致的 pos_embed / decoder_pos_embed 参数容器（方便后续 load）=====
    # encoder pos_embed: [1, L, C]，L = s3*s3，C = embed_dim[-1]
    C_enc = model.pos_embed.shape[-1]
    model.pos_embed = nn.Parameter(torch.zeros(1, s3*s3, C_enc), requires_grad=False)

    # decoder pos_embed: [1, L, Cdec]
    C_dec = model.decoder_pos_embed.shape[-1]
    model.decoder_pos_embed = nn.Parameter(torch.zeros(1, s3*s3, C_dec), requires_grad=False)

    # ===== 4) 对 ckpt 中的 pos_embed / decoder_pos_embed 做插值到 s3×s3 =====
    if "pos_embed" in state:
        state["pos_embed"] = _interp_pos_embed(state["pos_embed"], s3)
    if "decoder_pos_embed" in state:
        state["decoder_pos_embed"] = _interp_pos_embed(state["decoder_pos_embed"], s3)

    # ===== 5) 加载权重 =====
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[InfMAE] Loaded (resize from 224 -> {H}). Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if not strict and (len(missing) or len(unexpected)):
        # 可选：打印便于调试
        if len(missing) <= 30 and len(unexpected) <= 30:
            if missing:   print("  Missing:", missing)
            if unexpected:print("  Unexpected:", unexpected)

    # ===== 6) 保险：把 decoder 的位置编码也置入正确 device（上面 load 后一般已就位）=====
    model.pos_embed.requires_grad_(False)
    model.decoder_pos_embed.requires_grad_(False)



class InfMAE_UPerHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        features: int = 256,   # UPerNet里的统一通道数
        freeze_backbone: bool = False,
        target_input_size=512,
        infmae_kwargs: Optional[dict] = None,
        pretrain_path: Optional[str] = None,
    ):
        super().__init__()
        infmae_kwargs = infmae_kwargs or {}
        ppm_channels = features
        self.encoder = MaskedAutoencoderInfMAE(
            img_size=[512, 128, 64],      # 你之前已经改到 512 输入
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
            **infmae_kwargs
        )

        if pretrain_path is not None:
            load_infmae_pretrain(self.encoder, pretrain_path, target_input_size=target_input_size, strict=False)

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.backbone = InfMAEBackbone(self.encoder)

        # UPer 头：输入通道对应 4 级特征
        in_channels = [256, 384, 768, 768]
        self.head = UPerHead(
            in_channels=in_channels,
            channels=ppm_channels,
            num_classes=num_classes,
            ppm_pool_scales=(1, 2, 3, 6),
            align_corners=True,
        )

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        feats = self.backbone(x)               # 4 层: [1/4, 1/8, 1/16, 1/16]
        logits_low = self.head(feats)          # [B, C, H/4, W/4]
        logits = F.interpolate(logits_low, size=(H, W), mode='bilinear', align_corners=True)
        return logits


# ------------------------------
# 使用样例
# ------------------------------

if __name__ == "__main__":
    # 假设有一个 InfMAE 预训练权重
    ckpt = r"D:\codes\InfMAE-main\pre_weights\InfMAE_Inf30.pth"

    model = InfMAE_UPerHead(
        num_classes=1,
        features=256,
        freeze_backbone=True,
        pretrain_path=ckpt,  # 若无就设为 None
    )

    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)  # [2, 6, 224, 224]
    print("Output:", y.shape)

    # 检查可训练参数数量（应该只有 lora/gate/head）
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.3f}M / Total: {total/1e6:.3f}M")

