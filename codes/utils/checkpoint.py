# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch

def load_checkpoint(cfg, model, optimizer, scheduler, logger):
    logger.info(f"==============> Resuming form {cfg.train.resume_path}....................")
    checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage.cuda())
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(msg)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"]
    logger.info("==> loaded checkpoint from {}\n".format(cfg.train.resume_path) +
                "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))
    return start_epoch + 1


def save_checkpoint(cfg, epoch, model, optimizer, scheduler, logger, det_best=False, seg_best=False):
    save_state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'lr': optimizer.param_groups[0]["lr"]
    }

    # save last checkpoint
    last_checkpoint_path = os.path.join(cfg["train"]["output_dir"], f'last_checkpoint.pth')
    torch.save(save_state, last_checkpoint_path)
    
    # save best segmentation model
    if seg_best:
        seg_best_model_path = os.path.join(cfg["train"]["output_dir"], f'best_model.pth')
        torch.save(save_state, seg_best_model_path)