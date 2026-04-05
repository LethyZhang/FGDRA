import os
import argparse
import yaml
from datetime import datetime


def get_option():
    parser = argparse.ArgumentParser()

    # Main task: train / test / demo
    parser.add_argument(
        '-task',
        default='train',
        type=str,
        choices=['train', 'test', 'demo'],
        help='choose the task for running the model'
    )

    # Model task selector: add UIE support
    parser.add_argument(
        '-model_task',
        default='uie',
        type=str,
        choices=['uie'],  # ← 已加入 uie
        help='the model of the task'
    )

    parser.add_argument(
        '-device',
        default='cuda',
        type=str,
        help='choose the device to run the model'
    )

    opt = parser.parse_args()
    opt = opt_format(opt)
    return opt


def load_yaml(path):
    with open(path, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    return model_config


def save_yaml(path, file_dict):
    with open(path, 'w') as f:
        f.write(yaml.dump(file_dict, allow_unicode=True))


def opt_format(opt):
    # Root directory
    opt.root = os.getcwd()

    # Load config according to model task
    # config/<model_task>.yaml
    opt.config = r'{}/config/{}.yaml'.format(opt.root, opt.model_task)
    opt.config = load_yaml(opt.config)

    # Time stamp
    proper_time = str(datetime.now()).split('.')[0].replace(':', '-')

    # Experiment naming
    opt.config['exp_name'] = '{}_{}'.format(opt.task, opt.config['exp_name'])

    # Folder
    opt.experiments = r'{}/experiments/{}'.format(
        opt.root,
        '{} {}'.format(proper_time, opt.config['exp_name'])
    )
    if not os.path.exists(opt.experiments):
        os.makedirs(opt.experiments, exist_ok=True)

    # Save copied config
    config_path = r'{}/config.yaml'.format(opt.experiments)
    save_yaml(config_path, opt.config)

    # Determine save directories
    if opt.task == 'demo' or (opt.task == 'test' and opt.config['test']['save'] != False):
        opt.save_image = True
        opt.save_image_dir = r'{}/{}'.format(opt.experiments, 'images')
        if not os.path.exists(opt.save_image_dir):
            os.makedirs(opt.save_image_dir, exist_ok=True)

    # Logger
    opt.log_path = r'{}/logger.log'.format(opt.experiments)

    # Model saving path
    if opt.task == 'train':
        opt.save_model = True
        opt.save_model_dir = r'{}/{}'.format(opt.experiments, 'models')
        if not os.path.exists(opt.save_model_dir):
            os.makedirs(opt.save_model_dir, exist_ok=True)

    return opt
