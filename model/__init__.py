import torch
from importlib import import_module

from .uie import FGDRAUIENet, FGDRAUIENetS  # ← 新增

__all__ = {
    'FGDRAUIENet',
    'FGDRAUIENetS',
    'import_model'
}


def import_model(opt):
    # Construct the model class name
    # model_task: lle / isp / uie / sr
    model_name = 'FGDRA' + opt.model_task.upper()  # e.g. FGDRAUIE

    kwargs = {'channels': opt.config['model']['channels']}

    # Choose training-time or inference-time model
    if opt.config['model']['type'] == 're-parameterized':
        model_name += 'NetS'  # e.g. FGDRAUIENetS
    elif opt.config['model']['type'] == 'original':
        model_name += 'Net'  # e.g. FGDRAUIENet
        kwargs['rep_scale'] = opt.config['model']['rep_scale']
    else:
        raise ValueError('unknown model type, choose from [original, re-parameterized]')

    # Dynamically load class from model module
    ModelClass = getattr(import_module('model'), model_name)

    model = ModelClass(**kwargs).to(opt.device)

    # Pretrained weight loading
    if opt.config['model']['pretrained']:
        model.load_state_dict(torch.load(opt.config['model']['pretrained']), strict=False)

    # If need slim
    if opt.config['model']['type'] == 'original' and opt.config['model']['need_slim'] is True:
        model = model.slim().to(opt.device)

    return model
