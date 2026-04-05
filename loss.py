import torch
import torch.nn as nn
from option import get_option


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, inp, target):
        return ((nn.functional.mse_loss(inp, target, reduction='none') + self.eps2) ** .5).mean()


#####################################################################################################
class OutlierAwareLoss(nn.Module):
    def __init__(self, ):
        super(OutlierAwareLoss, self).__init__()

    def forward(self, out, lab):
        delta = out - lab
        var = delta.std((2, 3), keepdims=True) / (2 ** .5)
        avg = delta.mean((2, 3), True)
        weight = torch.tanh((delta - avg).abs() / (var + 1e-6)).detach()
        loss = (delta.abs() * weight).mean()
        return loss


#####################################################################################################
class LossWarmup(nn.Module):
    def __init__(self):
        super(LossWarmup, self).__init__()
        self.loss_cb = CharbonnierLoss(1e-8)
        self.loss_cs = nn.CosineSimilarity()

    def forward(self, inp, gt, warmup1, warmup2):
        loss = self.loss_cb(warmup2, inp) + \
               (self.loss_cb(warmup1, gt) + (1 - self.loss_cs(warmup1.clip(0, 1), gt)).mean())

        return loss

class LossUIE(nn.Module):
    def __init__(self):
        super(LossUIE, self).__init__()
        self.loss_oa = OutlierAwareLoss()
        self.loss_cs = nn.CosineSimilarity()
        self.psnr = PSNRLoss()

    def forward(self, out, gt):
        loss = self.loss_oa(out, gt) \
             + (1 - self.loss_cs(out.clip(0,1), gt)).mean() \
             + 2 * self.psnr(out, gt)
        return loss


class LossLLE(nn.Module):
    def __init__(self):
        super(LossLLE, self).__init__()
        self.loss_cs = nn.CosineSimilarity()
        self.loss_oa = OutlierAwareLoss()
        self.psnr = PSNRLoss()

    def forward(self, out, gt):
        loss = (self.loss_oa(out, gt) + (1 - self.loss_cs(out.clip(0, 1), gt)).mean()) + 2 * self.psnr(out, gt)
        return loss


class LossISP(nn.Module):
    def __init__(self):
        super(LossISP, self).__init__()
        self.loss_cs = nn.CosineSimilarity()
        self.loss_oa = OutlierAwareLoss()
        self.psnr = PSNRLoss()

    def forward(self, out, gt):
        loss = (self.loss_oa(out, gt) + (1 - self.loss_cs(out.clip(0, 1), gt)).mean()) + 2 * self.psnr(out, gt)
        return loss


def import_loss(training_task):
    if training_task == 'isp':
        return LossISP()
    if training_task == 'uie':
        return LossUIE()
    elif training_task == 'lle':
        return LossLLE()
    elif training_task == 'warmup':
        return LossWarmup()
    else:
        raise ValueError('unknown training task, please choose from [isp, lle, uie, warmup].')


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        imdff = pred - target
        rmse = ((imdff ** 2).mean(dim=(1, 2, 3)) + 1e-8).sqrt()
        loss = 20 * torch.log10(1 / rmse).mean()
        loss = (50.0 - loss) / 100.0
        return loss
