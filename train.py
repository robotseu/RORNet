import os
import time
import datetime
import argparse
import numpy as np
import yaml
import random
import torch

import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from codes.scheduler.build import build_lr_scheduler
from codes.utils.logger import create_logger
from codes.utils.metric import AverageMeter
from codes.utils.checkpoint import save_checkpoint
from codes.models.model import TacGrasp_Net
from codes.dataset.dataset import TacGraspDataSet
from eval import validate

import logging
logging.getLogger('PIL').setLevel(logging.WARNING)


def train_one_epoch(cfg, model, optimizer, scheduler, data_loader, writer, epoch):
    model.train()
    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')

    start = time.time()
    end = time.time()
    for idx, (ref, img_seq, tac_seq, label) in enumerate(data_loader):

        ref = ref.cuda(non_blocking=True)
        img_seq = img_seq.cuda(non_blocking=True)
        tac_seq = tac_seq.cuda(non_blocking=True)

        logits_per_image, logits_per_text, loss = model(ref, img_seq, tac_seq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 损失更新
        losses.update(loss.item(), img_seq.size(0))

        # 记录tensorboard损失曲线
        global_step = epoch * num_iters + idx
        writer.add_scalar("loss/train", losses.avg, global_step=global_step)

        if idx % cfg["train"]["log_period"] == 0 or idx == len(data_loader):
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{cfg["train"]["epochs"]}][{idx}/{num_iters}]  '
                f'lr {lr:.7f}  '
                f'Time {batch_time.val:.4f}  '
                f'Loss {losses.avg:.4f}  '
            )

        batch_time.update(time.time() - end)
        end = time.time()

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def main(cfg):

    # 加载模型
    model = TacGrasp_Net(cfg).cuda()

    # model.load_state_dict(torch.load("/home/lab404/Expand1/zzy/vlt_grasp/tac-grasp-net/output/1221_1902_vivit/last_checkpoint.pth")["state_dict"])

    # 数据集读取
    cfg["dataset"]["split"] = "train"
    train_set = TacGraspDataSet(cfg, model.preprocess)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg["train"]["batch_size"],
                              shuffle=True,
                              num_workers=cfg["train"]["num_workers"])

    cfg["dataset"]["split"] = "test"
    val_set = TacGraspDataSet(cfg, model.preprocess)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=cfg["train"]["batch_size"],
                            shuffle=False,
                            num_workers=cfg["train"]["num_workers"])

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["optim"]["lr"],
                                 betas=cfg["optim"]["betas"],
                                 eps=float(cfg["optim"]["eps"]))


    # 打印模型信息
    total_params = sum([param.nelement() for param in model.parameters()])
    trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad is True])
    logger.info("Number of all params: %.2fM" % (total_params / 1e6))
    logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))

    # 设置学习率
    scheduler = build_lr_scheduler(cfg, optimizer, len(train_loader))

    # 设置tensorboard
    writer = SummaryWriter(log_dir=cfg["train"]["output_dir"])

    start_epoch = 0
    best_loss = 1e9

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_one_epoch(cfg, model, optimizer, scheduler, train_loader, writer, epoch)
        loss = validate(cfg, model, val_loader, writer, epoch, logger)

        # save checkpoints
        if epoch % cfg["train"]["save_period"] == 0 or epoch == (cfg["train"]["epochs"] - 1):
            logger.info(f"saving checkpoints ...")
            save_checkpoint(cfg, epoch, model, optimizer, scheduler, logger)
            if loss < best_loss:
                save_checkpoint(cfg, epoch, model, optimizer, scheduler, logger, seg_best=True)
                best_loss = loss
            logger.info(f"checkpoints saved !!!\n")


if __name__ == '__main__':
    # 配置文件读取
    parser = argparse.ArgumentParser(description="PreGrasp_Net")
    args = parser.parse_args()
    with open('./config.yaml', encoding='utf-8') as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)

    # 设置随机种子
    if cfg["env"]["deterministic"]:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        cudnn.deterministic = True
        cudnn.benchmark = False

    torch.cuda.set_device(0)

    # 保存路径设置
    output_dir = cfg["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=cfg["train"]["output_dir"])

    # 主程序
    main(cfg)
