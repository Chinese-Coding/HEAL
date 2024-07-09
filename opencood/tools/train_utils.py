# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import glob
import importlib
from typing import overload

import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim
from torch import nn

from opencood.logger import get_logger

logger = get_logger()


def backup_script(full_path, folders_to_save=None):
    if folders_to_save is None:
        folders_to_save = ["models", "data_utils", "utils", "loss"]

    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)


def check_missing_key(model_state_dict: dict, ckpt_state_dict: dict):
    """
    用于检查 model_state_dict (模型的) 和 ckpt_state_dict (检查点) 之间是否存在不匹配的键  (参数名称)
    :param model_state_dict:
    :param ckpt_state_dict:
    :return:
    """
    model_keys = set(model_state_dict.keys())
    checkpoint_keys = set(ckpt_state_dict.keys())

    # 丢失的 key 和多余的 key, 就是做两个集合的差值
    missing_keys = model_keys - checkpoint_keys
    extra_keys = checkpoint_keys - model_keys

    missing_key_modules = set([keyname.split('.')[0] for keyname in missing_keys])
    extra_key_modules = set([keyname.split('.')[0] for keyname in extra_keys])

    logger.success('Loading Checkpoint')
    if len(missing_key_modules) == 0 and len(extra_key_modules) == 0:
        return

    def set_to_string(l: set) -> str:
        return '  '.join(l)

    logger.success(f'Missing keys from ckpt: {set_to_string(missing_key_modules)}')

    logger.success(f'Extra keys from ckpt: {set_to_string(extra_key_modules)}')
    logger.info(set_to_string(extra_keys))

    logger.success('You can go to tools/train_utils.py to print the full missing key name!\n')


def load_saved_model(saved_path: str, model):
    """
    Load saved model if existed

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def find_last_checkpoint(save_dir: str):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    bestval_file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if bestval_file_list:  # 如果找到了最好结果, 则从最好结果开始训练
        assert len(bestval_file_list) == 1  # TODO: 最好的结果只能有一个?
        # TODO: 这里可能有 bug
        bestval_file = bestval_file_list[0]  # 这行代码看着有些多余, 实际上是为了增加可读性
        epoch = int(bestval_file.split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at"))

        logger.success(f'resuming best validation model at epoch {epoch}')
        loaded_state_dict = torch.load(bestval_file_list[0], map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)
        return epoch, model

    initial_epoch = find_last_checkpoint(saved_path)
    if initial_epoch > 0:
        logger.success(f'resuming by loading epoch {initial_epoch}')
        loaded_state_dict = torch.load(os.path.join(saved_path, f'net_epoch{initial_epoch}.pth'), map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)

    return initial_epoch, model


def setup_train(hypes: dict):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
                backup_script(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path


@overload
def create_model(backbone_name: str, backbone_config: dict) -> nn.Module:
    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls
            # TODO: 找到就直接 break?

    if model is None:
        logger.error(f'backbone not found in models folder. '
                     f'Please make sure you have a python file named {model_filename} '
                     f'and has a class called target_model_name ignoring upper/lower case')
        exit(0)
    instance = model(backbone_config)
    return instance


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls
            # TODO: 找到就直接 break?

    if model is None:
        logger.error(f'backbone not found in models folder. '
                     f'Please make sure you have a python file named {model_filename} '
                     f'and has a class called target_model_name ignoring upper/lower case')
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(model.parameters(), lr=method_dict['lr'], **method_dict['args'])
    else:
        return optimizer_method(model.parameters(), lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, init_epoch=None):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    for _ in range(last_epoch):
        scheduler.step()

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) or isinstance(inputs, str) or not hasattr(inputs, 'to'):
            return inputs
        return inputs.to(device, non_blocking=True)
