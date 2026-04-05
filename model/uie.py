import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .utils import (
    MBRConv5,
    MBRConv3,
    MBRConv1,
    DropBlock,
    FST,
    FSTS,
)

# =========================================================
# Deterministic 32x32 downsample (avg_pool2d, no adaptive)
# =========================================================
def downsample_to_target_avgpool(F: torch.Tensor, target: int = 32) -> torch.Tensor:
    """
    Deterministic downsample to (target, target) by avg_pool2d.
    If H/W not divisible by target, first interpolate to nearest divisible size.
    """
    _, _, H, W = F.shape

    kh = H // target
    kw = W // target

    # too small feature map: fallback (avoid crash)
    if kh == 0 or kw == 0:
        return F

    # if not divisible, resize to divisible then pool
    if (H % target) != 0 or (W % target) != 0:
        H2 = kh * target
        W2 = kw * target
        H2 = max(H2, target)
        W2 = max(W2, target)

        # bilinear interpolate is typically deterministic-friendly;
        # if your env still complains, switch mode="nearest"
        F = Fnn.interpolate(F, size=(H2, W2), mode="bilinear", align_corners=False)

        H, W = H2, W2
        kh = H // target
        kw = W // target

    return Fnn.avg_pool2d(F, kernel_size=(kh, kw), stride=(kh, kw))


class FGDRAUIENet(nn.Module):
    """
    Training network (MBRConv + FST) with fgdra attention.
    - FFT guidance computed on deterministic 32x32 downsample (avg_pool2d-based).
    - Frequency guidance ONLY used to generate attention.
    - Slim supported.
    """
    def __init__(self, channels, rep_scale=4, fft_size=32):
        super(FGDRAUIENet, self).__init__()
        self.channels = channels
        self.fft_size = fft_size

        # ----------------------------
        # Backbone (same as FGDRA style)
        # ----------------------------
        self.head = FST(
            nn.Sequential(
                MBRConv5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale)
            ),
            channels
        )

        self.body = FST(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            channels
        )

        # ----------------------------
        # fgdra components (train)
        # ----------------------------
        # channel attention: input 2C (GAP(F) + GAP(LF))
        self.fgdra_fca = MBRConv1(2 * channels, channels, rep_scale=rep_scale)
        # spatial attention: input 3 (max(F), avg(F), HF_map)
        self.fgdra_fgsa = MBRConv1(2, channels, rep_scale=rep_scale)

        # gate params (static, no dynamic route)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.ones(1))
        self.lam   = nn.Parameter(torch.tensor(0.5))

        # output head
        self.tail = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.tail_warm = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.drop = DropBlock(3)

    def _fgdra_attention(self, F: torch.Tensor) -> torch.Tensor:
        """
        fgdra attention generation using low-res FFT guidance.
        Return A: (B,C,H,W)
        """
        B, C, H, W = F.shape

        # 0) deterministic downsample ONLY for FFT guidance
        F_ds = downsample_to_target_avgpool(F, target=self.fft_size)  # ideally (B,C,32,32)

        # 1) FFT magnitude (low-res)
        freq = torch.fft.fft2(F_ds)
        freq_mag = torch.log1p(torch.abs(freq))



        M = freq_mag

        # 3) Spatial attention (full-res): max/avg from F + HF_map upsampled
        max_map, _ = torch.max(F, dim=1, keepdim=True)  # (B,1,H,W)
        avg_map = torch.mean(F, dim=1, keepdim=True)    # (B,1,H,W)


        fgdra_spatial_in = torch.cat([max_map, avg_map], dim=1)

        A_s = torch.sigmoid(self.fgdra_fgsa(fgdra_spatial_in))                # (B,C,H,W)

        # 4) Channel attention: GAP(F) + GAP(M)
        gap_F = torch.mean(F, dim=(2, 3), keepdim=True)  # (B,C,1,1)


        gap_M = torch.mean(M, dim=(2, 3), keepdim=True)
        gap_M = gap_M / (gap_M.mean(dim=1, keepdim=True) + 1e-6)

        fgdra_channel_in = torch.cat([gap_F, gap_M], dim=1)  # (B,2C,1,1)
        # (B,2C,1,1)
        A_c = torch.sigmoid(self.fgdra_fca(fgdra_channel_in))                 # (B,C,1,1)

        # 5) static gated fusion (no dynamic route)
        Ag, Al = A_c, A_s
        lam = torch.clamp(self.lam, 0.0, 1.0)

        A_lin = self.alpha * Ag + self.beta * Al
        A_int = Ag * Al
        A = (1.0 - lam) * A_lin + lam * A_int                               # (B,C,H,W)

        return A

    def forward(self, x):
        x0 = self.head(x)
        F = self.body(x0)                       # (B,C,H,W)

        A = self._fgdra_attention(F)             # (B,C,H,W)
        F_hat = A * F

        return self.tail(F_hat)

    def forward_warm(self, x):
        x = self.drop(x)
        x = self.head(x)
        x = self.body(x)
        return self.tail(x), self.tail_warm(x)

    def slim(self):
        """
        Re-parameterize to FGDRAUIENetS (fgdra inference).
        - MBRConv1/3/5 -> Conv2d
        - FST params copy
        - alpha/beta/lam copy (top-level)
        """
        net_slim = FGDRAUIENetS(self.channels, fft_size=self.fft_size)
        weight_slim = net_slim.state_dict()

        # 1) reparam MBRConv layers -> Conv2d
        for name, mod in self.named_modules():
            if isinstance(mod, (MBRConv1, MBRConv3, MBRConv5)):
                w, b = mod.slim()
                if f"{name}.weight" in weight_slim:
                    weight_slim[f"{name}.weight"] = w
                    weight_slim[f"{name}.bias"] = b

            elif isinstance(mod, FST):
                # 2) copy FST parameters (must exist in FSTS with same keys)
                if f"{name}.bias" in weight_slim:
                    weight_slim[f"{name}.bias"] = mod.bias
                if f"{name}.weight1" in weight_slim:
                    weight_slim[f"{name}.weight1"] = mod.weight1
                if f"{name}.weight2" in weight_slim:
                    weight_slim[f"{name}.weight2"] = mod.weight2

            elif isinstance(mod, nn.PReLU):
                # 3) copy PReLU
                if f"{name}.weight" in weight_slim:
                    weight_slim[f"{name}.weight"] = mod.weight

        # 4) copy gate params to top-level (match inference net names)
        if "alpha" in weight_slim:
            weight_slim["alpha"] = self.alpha.detach()
        if "beta" in weight_slim:
            weight_slim["beta"] = self.beta.detach()
        if "lam" in weight_slim:
            weight_slim["lam"] = self.lam.detach()

        net_slim.load_state_dict(weight_slim)
        return net_slim


class FGDRAUIENetS(nn.Module):
    """
    Inference network (Conv + FSTS) with the SAME fgdra attention logic.
    Must match FGDRAUIENet._fgdra_attention exactly (except conv implementations).
    """
    def __init__(self, channels, fft_size=32):
        super(FGDRAUIENetS, self).__init__()
        self.channels = channels
        self.fft_size = fft_size

        self.head = FSTS(
            nn.Sequential(
                nn.Conv2d(3, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            channels
        )

        self.body = FSTS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            channels
        )

        # fgdra convs (slim target)
        self.fgdra_fca = nn.Conv2d(2 * channels, channels, 1, 1)
        self.fgdra_fgsa = nn.Conv2d(2, channels, 1, 1)

        # gate params (loaded from slim)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.ones(1))
        self.lam   = nn.Parameter(torch.tensor(0.5))

        self.tail = nn.Conv2d(channels, 3, 3, 1, 1)

    def _fgdra_attention(self, F: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F.shape


        F_ds = downsample_to_target_avgpool(F, target=self.fft_size)


        freq = torch.fft.fft2(F_ds)
        freq_mag = torch.log1p(torch.abs(freq))


        M = freq_mag


        max_map, _ = torch.max(F, dim=1, keepdim=True)
        avg_map = torch.mean(F, dim=1, keepdim=True)

        fgdra_spatial_in = torch.cat([max_map, avg_map], dim=1)

        A_s = torch.sigmoid(self.fgdra_fgsa(fgdra_spatial_in))


        gap_F = torch.mean(F, dim=(2, 3), keepdim=True)

        gap_M = torch.mean(M, dim=(2, 3), keepdim=True)
        gap_M = gap_M / (gap_M.mean(dim=1, keepdim=True) + 1e-6)

        fgdra_channel_in = torch.cat([gap_F, gap_M], dim=1)

        A_c = torch.sigmoid(self.fgdra_fca(fgdra_channel_in))


        Ag, Al = A_c, A_s
        lam = torch.clamp(self.lam, 0.0, 1.0)

        A_lin = self.alpha * Ag + self.beta * Al
        A_int = Ag * Al
        A = (1.0 - lam) * A_lin + lam * A_int

        return A

    def forward(self, x):
        x0 = self.head(x)
        F = self.body(x0)

        A = self._fgdra_attention(F)
        F_hat = A * F

        return self.tail(F_hat)
