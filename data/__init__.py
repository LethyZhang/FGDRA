# data/import_loader.py

from torch.utils import data
from importlib import import_module

# 原始任务的数据集

from .uiedata import UIEDataTrain, UIEDataValid, UIEDataTest   # ← 新增


__all__ = {
    'UIEDataTrain',
    'UIEDataValid',
    'UIEDataTest',
    'import_loader'
}


def import_loader(opt):
    """
    通用数据加载函数
    根据 opt.task (train/test/demo)
    以及 opt.model_task (uie/lle/isp/sr)
    自动选择合适的数据集。
    """

    model_task = opt.model_task.lower()

    # ============================================
    #                TRAIN MODE
    # ============================================
    if opt.task == 'train':
        train_inp = opt.config['train']['train_inp']
        train_gt  = opt.config['train']['train_gt']
        valid_inp = opt.config['train']['valid_inp']
        valid_gt  = opt.config['train']['valid_gt']

        # -------------------------
        #       UIE 特殊处理
        # -------------------------
        if model_task == 'uie':
            train_data = UIEDataTrain(opt, train_inp, train_gt, patch=256)
            valid_data = UIEDataValid(opt, valid_inp, valid_gt)

        # -------------------------
        #       原始任务
        # -------------------------
        elif model_task == 'lle':
            LLEDataClass = getattr(import_module('data'), 'LLEData')
            train_data = LLEDataClass(opt, train_inp, train_gt)
            valid_data = LLEDataClass(opt, valid_inp, valid_gt)

        elif model_task == 'isp':
            ISPDataClass = getattr(import_module('data'), 'ISPData')
            train_data = ISPDataClass(opt, train_inp, train_gt)
            valid_data = ISPDataClass(opt, valid_inp, valid_gt)

        elif model_task == 'sr':
            SRDataClass = getattr(import_module('data'), 'SRData')
            train_data = SRDataClass(opt, train_inp, train_gt)
            valid_data = SRDataClass(opt, valid_inp, valid_gt, 'valid')

        else:
            raise ValueError(f"Unsupported model_task: {model_task}")

        # build loaders
        train_loader = data.DataLoader(
            train_data,
            batch_size=opt.config['train']['batch_size'],
            shuffle=True,
            num_workers=opt.config['train']['num_workers'],
            drop_last=True,
        )

        valid_loader = data.DataLoader(
            valid_data,
            batch_size=1,
            shuffle=False,
            num_workers=opt.config['train']['num_workers'],
            drop_last=False,
        )

        return train_loader, valid_loader

    # ============================================
    #                TEST MODE
    # ============================================
    elif opt.task == 'test':
        test_inp = opt.config['test']['test_inp']
        test_gt  = opt.config['test']['test_gt']

        if model_task == 'uie':
            test_data = UIEDataTest(opt, test_inp, test_gt)

        elif model_task == 'lle':
            LLEDataClass = getattr(import_module('data'), 'LLEData')
            test_data = LLEDataClass(opt, test_inp, test_gt)

        elif model_task == 'isp':
            ISPDataClass = getattr(import_module('data'), 'ISPData')
            test_data = ISPDataClass(opt, test_inp, test_gt)

        elif model_task == 'sr':
            SRDataClass = getattr(import_module('data'), 'SRData')
            test_data = SRDataClass(opt, test_inp, test_gt, 'valid')

        else:
            raise ValueError(f"Unsupported model_task: {model_task}")

        test_loader = data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=opt.config['test']['num_workers'],
            drop_last=False,
        )

        return test_loader

    # ============================================
    #                DEMO MODE
    # ============================================
    elif opt.task == 'demo':
        demo_inp = opt.config['demo']['demo_inp']

        if model_task == 'uie':
            test_data = UIEDataValid(opt, demo_inp, None)

        elif model_task == 'lle':
            LLEDataClass = getattr(import_module('data'), 'LLEData')
            test_data = LLEDataClass(opt, demo_inp)

        elif model_task == 'isp':
            ISPDataClass = getattr(import_module('data'), 'ISPData')
            test_data = ISPDataClass(opt, demo_inp)

        elif model_task == 'sr':
            SRDataClass = getattr(import_module('data'), 'SRData')
            test_data = SRDataClass(opt, demo_inp)

        else:
            raise ValueError(f"Unsupported model_task: {model_task}")

        demo_loader = data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=opt.config['demo']['num_workers'],
            drop_last=False,
        )

        return demo_loader

    # ============================================
    #                ERROR
    # ============================================
    else:
        raise ValueError('unknown task, please choose from [train, test, demo]')
