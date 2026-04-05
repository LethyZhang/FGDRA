# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def _make_1d_dct_basis(n: int):

    x = torch.arange(n).float()
    k = torch.arange(n).float().unsqueeze(1)  # [n,1]
    basis = torch.cos(math.pi * (x + 0.5) * k / n)  # [n,n]
    return basis


def _make_2d_dct_bank(ksize: int, K: int):

    assert ksize in (3, 5)
    B = _make_1d_dct_basis(ksize)  # [ksize, ksize]

    pairs = []
    for u in range(ksize):
        for v in range(ksize):
            if not (u == 0 and v == 0):  # 排除 DC
                pairs.append((u, v))
    pairs = pairs[:K]
    bank = []
    for (u, v) in pairs:
        k2d = torch.ger(B[u], B[v])  # [ksize, ksize]
        bank.append(k2d)
    if len(bank) < K:

        for _ in range(K - len(bank)):
            bank.append(torch.ger(B[0], B[1]))
    bank = torch.stack(bank, dim=0)  # [K, ksize, ksize]
    return bank


def _zero_mean_unit_norm(k: torch.Tensor, eps: float = 1e-6):

    k = k - k.mean(dim=(-2, -1), keepdim=True)
    n = torch.linalg.vector_norm(k, ord=2, dim=(-2, -1), keepdim=True)
    k = k / (n + eps)
    return k


def _center_to_3x3(kernel_ksize: int, k: torch.Tensor):

    if kernel_ksize == 3:
        return k
    elif kernel_ksize == 5:
        return k[..., 1:4, 1:4]  # 中心裁剪
    else:
        raise ValueError(" ")



class DCTLinearBranch(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, K: int = 2, ksize: int = 3, use_gate: bool = True):
        super().__init__()
        # ★ 强制 DCT 核为 3×3，保证与 3×3 等效折叠的一致性
        assert ksize == 3
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.K = int(K)
        self.ksize = 3


        self.dw = nn.Conv2d(self.in_channels, self.in_channels * self.K, kernel_size=self.ksize,
                            padding=1, groups=self.in_channels, bias=False)


        self.pw = nn.Conv2d(self.in_channels * self.K, self.out_channels, kernel_size=1, bias=False)


        self.use_gate = use_gate
        if self.use_gate:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("gamma_dummy", torch.tensor(1.0))


        with torch.no_grad():
            k_bank = _make_2d_dct_bank(self.ksize, self.K)                  # [K, 3, 3]
            k_bank = _zero_mean_unit_norm(k_bank)                            # 归一化
            k_bank = k_bank.unsqueeze(0).repeat(self.in_channels, 1, 1, 1)   # [C, K, 3, 3]
            k_bank = k_bank.reshape(self.in_channels * self.K, 1, 3, 3)
            self.dw.weight.copy_(k_bank)

        for p in self.dw.parameters():
            p.requires_grad = False


        nn.init.zeros_(self.pw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)             # [B, C*K, H, W]
        y = self.pw(y)             # [B, O,   H, W]
        if self.use_gate:
            y = y * torch.tanh(self.gamma)
        return y

    def get_equivalent_3x3_weight(self) -> torch.Tensor:

        device = self.pw.weight.device
        dtype = self.pw.weight.dtype

        DW = self.dw.weight.detach()                 # [C*K, 1, 3, 3]
        CK = DW.shape[0]
        K = self.K
        assert CK % K == 0
        C = CK // K
        O = self.pw.weight.shape[0]

        DW_3x3 = DW.view(C, K, 3, 3)                 # [C,K,3,3]
        PW = self.pw.weight.view(O, C, K)            # [O,C,K]

        W = torch.einsum('ock,ckhw->ochw', PW, DW_3x3)  # [O,C,3,3]
        if self.use_gate:
            W = W * torch.tanh(self.gamma).to(dtype=dtype, device=device)
        return W

    def get_equivalent_bias(self) -> torch.Tensor:

        device = self.pw.weight.device
        dtype = self.pw.weight.dtype
        return torch.zeros(self.pw.weight.shape[0], device=device, dtype=dtype)



class DCTAddIntoConv3x3(nn.Module):

    def __init__(self, C_in: int, C_out: int, K: int = 2, ksize: int = 3, **kwargs):
        super().__init__()
        assert ksize in (3, 5)
        self.C_in = int(C_in)
        self.C_out = int(C_out)
        self.K = int(K)
        self.ksize = int(ksize)

        self.coeff = nn.Parameter(torch.zeros(self.C_out, self.C_in, self.K))  # 零初始化
        self.register_buffer("DW_bank", self._build_dw_bank())  # [C_in, K, 3, 3]
        self.gamma = nn.Parameter(torch.zeros(1))  # 标量门控

    def _build_dw_bank(self) -> torch.Tensor:

        with torch.no_grad():
            bank = _make_2d_dct_bank(self.ksize, self.K)   # [K,k,k]
            bank = _zero_mean_unit_norm(bank)
            bank = _center_to_3x3(self.ksize, bank)        # [K,3,3]
            bank = bank.unsqueeze(0).repeat(self.C_in, 1, 1, 1)  # [C_in, K, 3,3]
        return bank

    def get_equivalent_weight(self, in_channels: int = None) -> torch.Tensor:
        a = self.coeff  # [O,C,K]
        DW = self.DW_bank  # [C,K,3,3]
        W = torch.einsum('ock,ckhw->ochw', a, DW)  # [O,C,3,3]
        W = W * torch.tanh(self.gamma)
        return W

    def forward(self, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
        W_eff = W + self.get_equivalent_weight(self.C_in)
        return F.conv2d(x, W_eff, b, stride=1, padding=1, dilation=1, groups=1)



class MBRConv5(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv2_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)

    def forward(self, inp):

        x1 = self.conv(inp)
        x2 = self.conv1(inp)
        x3 = self.conv2(inp)
        x4 = self.conv_crossh(inp)
        x5 = self.conv_crossv(inp)
        x = torch.cat(
            [x1, x2, x3, x4, x5,
             self.conv_bn(x1),
             self.conv1_bn(x2),
             self.conv2_bn(x3),
             self.conv_crossh_bn(x4),
             self.conv_crossv_bn(x5)],
            1
        )
        out = self.conv_out(x)
        return out

    def slim(self):

        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        conv1_weight = self.conv1.weight
        conv1_bias = self.conv1.bias
        conv1_weight = F.pad(conv1_weight, (2, 2, 2, 2))  # 1x1 -> 5x5

        conv2_weight = self.conv2.weight
        conv2_weight = F.pad(conv2_weight, (1, 1, 1, 1))  # 3x3 -> 5x5
        conv2_bias = self.conv2.bias

        conv_crossv_weight = self.conv_crossv.weight
        conv_crossv_weight = F.pad(conv_crossv_weight, (1, 1, 2, 2))  # 1x3 -> 5x5
        conv_crossv_bias = self.conv_crossv.bias

        conv_crossh_weight = self.conv_crossh.weight
        conv_crossh_weight = F.pad(conv_crossh_weight, (2, 2, 1, 1))  # 3x1 -> 5x5
        conv_crossh_bias = self.conv_crossh.bias

        conv1_bn_weight = self.conv1.weight
        conv1_bn_weight = F.pad(conv1_bn_weight, (2, 2, 2, 2))  # 1x1 -> 5x5

        conv2_bn_weight = self.conv2.weight
        conv2_bn_weight = F.pad(conv2_bn_weight, (1, 1, 1, 1))  # 3x3 -> 5x5

        conv_crossv_bn_weight = self.conv_crossv.weight
        conv_crossv_bn_weight = F.pad(conv_crossv_bn_weight, (1, 1, 2, 2))  # 1x3 -> 5x5

        conv_crossh_bn_weight = self.conv_crossh.weight
        conv_crossh_bn_weight = F.pad(conv_crossh_bn_weight, (2, 2, 1, 1))  # 3x1 -> 5x5


        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        bn = self.conv1_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv1_bn_weight = self.conv1.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = F.pad(conv1_bn_weight, (2, 2, 2, 2))  # pad 到 5x5
        conv1_bn_bias = self.conv1.bias * k + b
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

        bn = self.conv2_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv2_bn_weight = self.conv2.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_weight = conv2_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_weight = F.pad(conv2_bn_weight, (1, 1, 1, 1))  # pad 到 5x5
        conv2_bn_bias = self.conv2.bias * k + b
        conv2_bn_bias = conv2_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = F.pad(conv_crossv_bn_weight, (1, 1, 2, 2))  # 1x3 -> 5x5
        conv_crossv_bn_bias = self.conv_crossv.bias * k + b
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossh_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = F.pad(conv_crossh_bn_weight, (2, 2, 1, 1))  # 3x1 -> 5x5
        conv_crossh_bn_bias = self.conv_crossh.bias * k + b
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias


        weight_all = torch.cat(
            [conv_weight, conv1_weight, conv2_weight,
             conv_crossh_weight, conv_crossv_weight,
             conv_bn_weight, conv1_bn_weight, conv2_bn_weight,
             conv_crossh_bn_weight, conv_crossv_bn_weight],
            0
        )  # [SumOrep, C, 5, 5]
        bias_all = torch.cat(
            [conv_bias, conv1_bias, conv2_bias,
             conv_crossh_bias, conv_crossv_bias,
             conv_bn_bias, conv1_bn_bias, conv2_bn_bias,
             conv_crossh_bn_bias, conv_crossv_bn_bias],
            0
        )  # [SumOrep]

        weight_compress = self.conv_out.weight.squeeze()  # [O, SumOrep]

        W_flat = weight_all.view(weight_all.size(0), -1)                 # [SumOrep, C*5*5]
        W_eff_flat = torch.matmul(weight_compress, W_flat)               # [O, C*5*5]
        weight = W_eff_flat.view(self.conv_out.out_channels, self.in_channels, 5, 5)

        bias = torch.matmul(weight_compress, bias_all)                   # [O]
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias



class MBRConv3(nn.Module):

    def __init__(self, in_channels, out_channels, rep_scale=4, use_dct=True, dct_K=4, dct_ksize=3):
        super(MBRConv3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        self.use_dct = use_dct


        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))


        if self.use_dct:

            dct_ksize = 3
            self.dct_branch = DCTLinearBranch(
                in_channels=in_channels,
                out_channels=out_channels * rep_scale,
                K=dct_K, ksize=dct_ksize, use_gate=True
            )

            self.conv_out = nn.Conv2d(out_channels * rep_scale * 9, out_channels, 1)
        else:
            self.dct_branch = None
            self.conv_out = nn.Conv2d(out_channels * rep_scale * 8, out_channels, 1)

    def forward(self, inp):

        x0 = self.conv(inp)
        x1 = self.conv1(inp)
        x2 = self.conv_crossh(inp)
        x3 = self.conv_crossv(inp)

        feats = [
            x0, x1, x2, x3,
            self.conv_bn(x0),
            self.conv1_bn(x1),
            self.conv_crossh_bn(x2),
            self.conv_crossv_bn(x3)
        ]


        if self.dct_branch is not None:
            x_dct = self.dct_branch(inp)
            feats.append(x_dct)

        x = torch.cat(feats, dim=1)
        out = self.conv_out(x)
        return out

    def slim(self):

        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        conv1_weight = F.pad(self.conv1.weight, (1, 1, 1, 1))
        conv1_bias = self.conv1.bias

        conv_crossh_weight = F.pad(self.conv_crossh.weight, (1, 1, 0, 0))
        conv_crossh_bias = self.conv_crossh.bias

        conv_crossv_weight = F.pad(self.conv_crossv.weight, (0, 0, 1, 1))
        conv_crossv_bias = self.conv_crossv.bias


        def fuse_bn(conv_w, conv_b, bn_layer):
            bn = bn_layer[0]
            k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w_fused = conv_w * k.view(-1, 1, 1, 1)
            b_fused = (conv_b - bn.running_mean) * k + bn.bias
            return w_fused, b_fused

        conv_bn_weight, conv_bn_bias = fuse_bn(self.conv.weight, self.conv.bias, self.conv_bn)
        conv_bn_weight = conv_bn_weight  # already 3×3

        conv1_bn_weight, conv1_bn_bias = fuse_bn(self.conv1.weight, self.conv1.bias, self.conv1_bn)
        conv1_bn_weight = F.pad(conv1_bn_weight, (1, 1, 1, 1))

        conv_crossh_bn_weight, conv_crossh_bn_bias = fuse_bn(self.conv_crossh.weight, self.conv_crossh.bias,
                                                             self.conv_crossh_bn)
        conv_crossh_bn_weight = F.pad(conv_crossh_bn_weight, (1, 1, 0, 0))

        conv_crossv_bn_weight, conv_crossv_bn_bias = fuse_bn(self.conv_crossv.weight, self.conv_crossv.bias,
                                                             self.conv_crossv_bn)
        conv_crossv_bn_weight = F.pad(conv_crossv_bn_weight, (0, 0, 1, 1))


        if self.dct_branch is not None:
            dct_w = self.dct_branch.get_equivalent_3x3_weight()  # [Orep, C, 3,3]
            dct_b = self.dct_branch.get_equivalent_bias()
        else:
            dct_w = None
            dct_b = None


        weight_list = [
            conv_weight, conv1_weight, conv_crossh_weight, conv_crossv_weight,
            conv_bn_weight, conv1_bn_weight, conv_crossh_bn_weight, conv_crossv_bn_weight
        ]
        bias_list = [
            conv_bias, conv1_bias, conv_crossh_bias, conv_crossv_bias,
            conv_bn_bias, conv1_bn_bias, conv_crossh_bn_bias, conv_crossv_bn_bias
        ]

        if dct_w is not None:
            weight_list.append(dct_w)
            bias_list.append(dct_b)

        weight_all = torch.cat(weight_list, dim=0)  # [SumOrep, C, 3,3]
        bias_all = torch.cat(bias_list, dim=0)  # [SumOrep]


        weight_compress = self.conv_out.weight.view(self.conv_out.out_channels, -1)  # [O, SumOrep]

        W_flat = weight_all.view(weight_all.size(0), -1)  # [SumOrep, C*3*3]
        W_eff = torch.matmul(weight_compress, W_flat)  # [O, C*3*3]
        W_eff = W_eff.view(self.conv_out.out_channels, self.in_channels, 3, 3)

        b_eff = torch.matmul(weight_compress, bias_all)  # [O]
        if self.conv_out.bias is not None:
            b_eff = b_eff + self.conv_out.bias

        return W_eff, b_eff



class MBRConv1(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def forward(self, inp):

        x0 = self.conv(inp)
        x = torch.cat([x0, self.conv_bn(x0)], 1)
        out = self.conv_out(x)
        return out

    def slim(self):

        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        weight_all = torch.cat([conv_weight, conv_bn_weight], 0)  # [2*Orep, C, 1, 1]
        bias_all = torch.cat([conv_bias, conv_bn_bias], 0)        # [2*Orep]

        weight_compress = self.conv_out.weight.squeeze()          # [O, 2*Orep]
        W_flat = weight_all.view(weight_all.size(0), -1)          # [2*Orep, C*1*1] = [2*Orep, C]
        W_eff_flat = torch.matmul(weight_compress, W_flat)        # [O, C]
        weight = W_eff_flat.view(self.conv_out.out_channels, self.in_channels, 1, 1)

        bias = torch.matmul(weight_compress, bias_all)            # [O]
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias



class FST(nn.Module):
    def __init__(self, block1, channels):
        super(FST, self).__init__()
        self.block1 = block1
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias


class FSTS(nn.Module):
    def __init__(self, block1, channels):
        super(FSTS, self).__init__()
        self.block1 = block1
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias


class DropBlock(nn.Module):
    def __init__(self, block_size, p=0.5):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.p = p / block_size / block_size

    def forward(self, x):
        mask = 1 - (torch.rand_like(x[:, :1]) >= self.p).float()
        mask = nn.functional.max_pool2d(mask, self.block_size, 1, self.block_size // 2)
        return x * (1 - mask)
