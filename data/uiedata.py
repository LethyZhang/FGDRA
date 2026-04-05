# data/uiedata.py

import os
import torch
import numpy as np
from PIL import Image


# ============================================================
#  UIE Train Dataset（与 MobileIE 原版完全一致：随机裁剪 + 内部 to CUDA）
# ============================================================
class UIEDataTrain(torch.utils.data.Dataset):
    def __init__(self, opt, inp_path, gt_path, patch=256):
        self.files = sorted(os.listdir(inp_path))
        self.inp_path = inp_path
        self.gt_path = gt_path
        self.patch = patch
        self.device = opt.device

    def __getitem__(self, index):
        fname = self.files[index]

        inp = Image.open(os.path.join(self.inp_path, fname)).convert('RGB')
        gt  = Image.open(os.path.join(self.gt_path,  fname)).convert('RGB')

        # ----------------------------
        # 预防 H < patch / W < patch
        # ----------------------------
        if inp.height < self.patch or inp.width < self.patch:
            new_w = max(inp.width,  self.patch)
            new_h = max(inp.height, self.patch)
            inp = inp.resize((new_w, new_h), Image.BILINEAR)
            gt  = gt.resize((new_w, new_h), Image.BILINEAR)

        # PIL → numpy float32
        inp = np.array(inp).astype(np.float32) / 255.
        gt  = np.array(gt ).astype(np.float32) / 255.

        H, W, _ = inp.shape
        p = self.patch

        # ----------------------------
        #     random crop 256×256
        # ----------------------------
        y = np.random.randint(0, H - p + 1)
        x = np.random.randint(0, W - p + 1)

        inp = inp[y:y+p, x:x+p]
        gt  = gt[y:y+p, x:x+p]

        # HWC → CHW
        inp = torch.from_numpy(inp.transpose(2, 0, 1))
        gt  = torch.from_numpy(gt.transpose(2, 0, 1))

        # ----------------------------
        # Dataset 内直接搬到 GPU
        # ----------------------------
        inp = inp.to(self.device, non_blocking=True)
        gt  = gt.to(self.device, non_blocking=True)

        return inp, gt, fname

    def __len__(self):
        return len(self.files)


# ============================================================
#  UIE Valid（原图尺寸，不裁剪，不 resize，自动搬到 GPU）
# ============================================================
class UIEDataValid(torch.utils.data.Dataset):
    def __init__(self, opt, inp_path, gt_path):
        self.files = sorted(os.listdir(inp_path))
        self.inp_path = inp_path
        self.gt_path = gt_path
        self.device = opt.device

    def __getitem__(self, index):
        fname = self.files[index]

        inp = Image.open(os.path.join(self.inp_path, fname)).convert('RGB')
        inp = torch.from_numpy(
            np.array(inp).astype(np.float32).transpose(2,0,1) / 255.
        ).to(self.device, non_blocking=True)

        gt = None
        if self.gt_path is not None:
            gt_img = Image.open(os.path.join(self.gt_path, fname)).convert('RGB')
            gt = torch.from_numpy(
                np.array(gt_img).astype(np.float32).transpose(2,0,1) / 255.
            ).to(self.device, non_blocking=True)

        return inp, gt, fname

    def __len__(self):
        return len(self.files)


# ============================================================
#  UIE Test（与 Valid 相同）
# ============================================================
class UIEDataTest(UIEDataValid):
    pass
