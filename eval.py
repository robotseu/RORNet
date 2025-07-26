import time
import torch
import torch.optim
from codes.utils.metric import AverageMeter

import logging
logging.getLogger('PIL').setLevel(logging.WARNING)


def validate(cfg, model, data_loader, writer, epoch, logger, prefix='Val'):

    model.eval()
    batch_time = AverageMeter('Time', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    num_iters = len(data_loader)

    with torch.no_grad():
        end = time.time()
        for idx, (ref, img_seq, tac_seq, label) in enumerate(data_loader):

            ref = ref.cuda(non_blocking=True)
            img_seq = img_seq.cuda(non_blocking=True)
            tac_seq = tac_seq.cuda(non_blocking=True)

            logits_per_image, logits_per_text, loss = model(ref, img_seq, tac_seq)

            # 损失更新
            losses.update(loss.item(), img_seq.size(0))

            # 记录tensorboard损失曲线
            global_step = epoch * num_iters + idx
            writer.add_scalar("loss/eval", losses.avg, global_step=global_step)

            if idx % cfg["train"]["log_period"] == 0 or idx == (len(data_loader) - 1):
                logger.info(
                    f'Evaluation on {prefix}: [{idx}/{len(data_loader)}]  '
                    f'Time {batch_time.val:.3f}  '
                    f'Loss {losses.avg:.4f}  '
                )
            batch_time.update(time.time() - end)
            end = time.time()

    return losses.avg