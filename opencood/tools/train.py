# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset, _build_dataset
from opencood.logger import get_logger
from opencood.tools import train_utils


def train_parser():
    """
    这段函数可以直接从命令行中读取参数, 而无需传参
    Returns:
    解析的命令行参数
    """
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True, help='data generation yaml file needed')
    parser.add_argument('--model_dir', default='', help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate", help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    logger.success('Dataset Building')
    # train_dataset = build_dataset(hypes, visualize=False, train=True)
    # validate_dataset = build_dataset(hypes, visualize=False, train=False)

    train_dataset = _build_dataset(hypes['fusion']['core_method'], hypes['fusion']['dataset'], hypes, False, True)
    validate_dataset = _build_dataset(hypes['fusion']['core_method'], hypes['fusion']['dataset'], hypes, False, False)

    logger.success(f'训练数据集类型: {type(train_dataset)}')
    logger.success(f'验证数据集类型: {type(validate_dataset)}')

    """
    在 PyTorch 的 DataLoader 中，collate_fn 参数用于定义如何将一个 batch 的样本组合在一起。
    通常情况下，DataLoader 会自动将一个 batch 中的样本堆叠成张量（tensor），但在某些情况下，可能需要自定义这种组合方式。这时，collate_fn 就显得非常有用。
    """
    train_loader = DataLoader(train_dataset, batch_size=hypes['train_params']['batch_size'],
                              num_workers=1, collate_fn=train_dataset.collate_batch_train,
                              shuffle=False, pin_memory=True, drop_last=True, prefetch_factor=2)

    val_loader = DataLoader(validate_dataset, batch_size=hypes['train_params']['batch_size'],
                            num_workers=1, collate_fn=train_dataset.collate_batch_train,
                            shuffle=False, pin_memory=True, drop_last=True, prefetch_factor=2)
    logger.success('数据集加载完毕, 开始创建模型')

    model = train_utils._create_model(hypes['model']['core_method'], hypes['model']['args'])

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        # scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        logger.success(f'resume from {init_epoch} epoch.')
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder to save the model,
        saved_path = train_utils.setup_train(hypes)
        # scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            absolute_gup_index = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[torch.cuda.current_device()])
            logger.success(f'Using device: {device}, Using GPU index: {absolute_gup_index}')
        else:
            logger.success(f'Using device: {device}')
        model.to(device)
    else:
        logger.error('cuda is not available. Please check.')
        exit(-1)
    logger.success(f'model 类型: {type(model)}')
    logger.success(f'loss 类型: {type(criterion)}')
    logger.success(f'optimizer 类型: {type(optimizer)}')

    # record training
    writer = SummaryWriter(saved_path)

    logger.success('Training start')
    epoch_times = []
    total_start_time = datetime.now()
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(train_dataset, "supervise_single") \
        else train_dataset.supervise_single

    # used to help schedule learning rate
    epoches = max(epoches, init_epoch)
    for epoch in range(init_epoch, epoches):
        logger.success(f'[{epoch}/{epoches}] epoch start training')
        epoch_start_time = datetime.now()
        for param_group in optimizer.param_groups:
            logger.success(f'learning rate {param_group["lr"]}')

        # the model will be evaluation mode during validation
        model.train()  # Sets the module in training mode
        try:  # heter_model stage2
            model.model_train_init()
        except AttributeError:
            # print("No model_train_init function")
            logger.error('No model_train_init function')

        train_loader_len = len(train_loader)
        pbar = tqdm(enumerate(train_loader), total=train_loader_len, desc=f"Epoch {epoch}")
        for i, batch_data in pbar:
            # TODO: 为什么这里需要加一个关于 object_bbx_mask 的判断?
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum() == 0:
                continue

            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            output_dict = model(batch_data['ego'])

            final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
            loss_dict = criterion.logging(epoch, i, train_loader_len, writer)

            if supervise_single_flag:
                final_loss += (criterion(output_dict, batch_data['ego']['label_dict_single'], suffix="_single") *
                               hypes['train_params'].get("single_weight", 1))
                loss_dict = criterion.logging(epoch, i, train_loader_len, writer, suffix="_single")

            # Update progress bar with current loss
            pbar.set_postfix({k: f'{v:.4f}' for k, v in loss_dict.items()})
            # back-propagation
            final_loss.backward()
            optimizer.step()

            # torch.cuda.empty_cache()  # it will destroy memory buffer
        if epoch % hypes['train_params']['save_freq'] == 0:
            logger.success(f'保存中继点： {epoch}')
            torch.save(model.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            logger.success('开始计算损失')
            valid_ave_losses = []
            """
            with torch.no_grad(): 是 PyTorch 中的一个上下文管理器，用于在某个代码块中暂时关闭自动求导机制。
            具体来说，它的作用是确保在这个代码块中的所有操作都不会记录梯度信息，从而节省显存和计算资源。
            这在进行模型推理（inference）或评估（evaluation）时特别有用，因为在这些情况下不需要计算梯度。
            """
            with torch.no_grad():
                pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validation Loss')
                for i, batch_data in pbar:
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    output_dict = model(batch_data['ego'])

                    final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
                    logger.info(f'val loss {final_loss:.3f}')
                    valid_ave_losses.append(final_loss.item())

                    # Update progress bar with current loss
                    pbar.set_postfix({'loss': f'{final_loss:.3f}'})

            valid_ave_loss = statistics.mean(valid_ave_losses)
            logger.success(f'At epoch {epoch}, the averaged validation loss is {valid_ave_loss}')
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:  # 如果这一轮的损失精度小于之前的最小的损失精度, 那么就保存当前的训练结果为最好的训练结果
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(), os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if (lowest_val_epoch != -1 and
                        os.path.exists(os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % lowest_val_epoch))):
                    os.remove(os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % lowest_val_epoch))
                lowest_val_epoch = epoch + 1

        during_time = (datetime.now() - epoch_start_time).total_seconds()
        epoch_times.append({epoch: during_time})

        logger.success(f'Epoch {epoch}, Total Time (train + validation): {during_time:.2f} seconds')
        # scheduler.step(epoch)
        logger.success(f'Dataset Building for {epoch + 1}')
        train_dataset.reinitialize()

    logger.success(f'Training Finished, checkpoints saved to {saved_path}')
    logger.success(f'Total train time: {(datetime.now() - total_start_time).total_seconds():.2f} seconds')
    logger.success(f'Eche epoch: {epoch_times}')

    run_test = True
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)


if __name__ == '__main__':
    logger = get_logger()
    main()
