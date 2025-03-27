import argparse
import os
import logging
import time
import datetime
import torch
import errno
import sys
from tensorboardX import SummaryWriter
from nas import cfg
from nas import build_dataset
from nas import make_lr_scheduler
from nas import make_optimizer
from nas import Checkpointer
from nas import MetricLogger
from nas import HSItrainnet
from nas import HSIsearchnet
from nas import model_visualize
import torch.nn.functional as F

ARCHITECTURES = {
    "searchnet": HSIsearchnet,
    "trainnet": HSItrainnet,
}


def build_model(cfga):
    meta_arch = ARCHITECTURES[cfga.MODEL.META_ARCHITECTURE]
    return meta_arch(cfga)


def setup_logger(name, save_dir, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class ResultAnalysis(object):
    def __init__(self):
        self.loss_sum = 0
        self.correct_pixel = 0
        self.total_pixel = 0

    def __call__(self, pred, targets):
        n, c, h, w = pred.size()
        logits = pred.permute(0, 2, 3, 1).contiguous().view(-1, c)
        labels = targets.view(-1)
        pred_label = torch.argmax(pred, dim=1)
        loss = F.cross_entropy(logits, labels, ignore_index=-1)
        correct_pixel = torch.eq(pred_label, targets).sum().item()
        label_mask = targets > -1
        labels_num = torch.sum(label_mask).item()
        self.loss_sum += float(loss * labels_num)
        self.correct_pixel += correct_pixel
        self.total_pixel += labels_num

    def reset(self):
        self.loss_sum = 0
        self.correct_pixel = 0
        self.total_pixel = 0

    def get_result(self):
        return self.correct_pixel / self.total_pixel * 100, self.loss_sum / self.total_pixel


@torch.inference_mode()
def inference(model, val_loaders):
    print('start_inference')
    model.eval()
    result_anal = ResultAnalysis()
    with torch.inference_mode():
        for images, targets in val_loaders:
            images = images.to(device='cuda', non_blocking=True)
            targets = targets.to(device='cuda', non_blocking=True)
            pred = model(images, targets)
            result_anal(pred, targets)
        acc, loss = result_anal.get_result()
        result_anal.reset()
    return acc, loss


def do_search(
        model,
        train_loaders,
        val_loaders,
        max_epoch,
        arch_start_epoch,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpointer_period,
        arguments,
        writer,
        cfgc,
        visual_dir):
    logger = logging.getLogger("nas.searcher")
    logger.info("Start searching")
    start_epoch = arguments["epoch"]
    start_training_time = time.time()
    val_best_acc, val_min_loss = 0, 10
    for epoch in range(start_epoch, max_epoch):
        model.reset_selection_cache()
        epoch = epoch + 1
        arguments["epoch"] = epoch
        train(model, train_loaders, optimizer, scheduler, epoch, train_arch=epoch > arch_start_epoch)
        if epoch > cfgc.SEARCH.ARCH_START_EPOCH:
            save_dir = os.path.join(visual_dir, 'visualize', f'arch_epoch{epoch}')
            model_visualize(save_dir)
        if epoch % val_period == 0:
            acc, loss = inference(model, val_loaders)
            if acc > val_best_acc:
                val_best_acc, val_min_loss = acc, loss
                checkpointer.save("model_best", **arguments)
                best_model_visualization_dir = os.path.join(visual_dir, 'visualize', 'best_model')
                model_visualize(best_model_visualization_dir)
            elif acc == val_best_acc and loss < val_min_loss:
                val_best_acc, val_min_loss = acc, loss
                checkpointer.save("model_best", **arguments)
                best_model_visualization_dir = os.path.join(visual_dir, 'visualize', 'best_model')
                model_visualize(best_model_visualization_dir)
            logger.info(f'val_acc: {acc:.2f}% val_loss: {loss:.4f}')
            writer.add_scalars('Search_acc', {'val_acc': acc}, epoch)
            writer.add_scalars('Search_loss', {'val_loss': loss}, epoch)
        if epoch % checkpointer_period == 0:
            checkpointer.save(f"model_{epoch:03d}", **arguments)
        if epoch == max_epoch:
            checkpointer.save("model_final", **arguments)
    total_training_time = time.time() - start_training_time
    logger.info(f"Total training time: {datetime.timedelta(seconds=total_training_time)}")


def train(model, data_loaders, optimizer, schedulers, epoch, train_arch=False):
    data_loader_w = data_loaders[0]
    data_loader_a = data_loaders[1]
    optim_w = optimizer['optim_w']
    optim_a = optimizer['optim_a']
    scheduler_w = schedulers['scheduler_w']
    scheduler_a = schedulers['scheduler_a']
    logger = logging.getLogger("nas.searcher")
    max_iter = len(data_loader_w)
    model.train()
    meters = MetricLogger(delimiter="  ")
    end = time.time()
    for iteration, (images, targets) in enumerate(data_loader_w):
        images = images.to(device='cuda', non_blocking=True)
        targets = targets.to(device='cuda', non_blocking=True)
        data_time = time.time() - end
        if train_arch:
            images_a, targets_a = next(iter(data_loader_a))
            images_a = images_a.to(device='cuda', non_blocking=True)
            targets_a = targets_a.to(device='cuda', non_blocking=True)
            loss = model(images_a, targets_a)
            optim_a.zero_grad()
            loss.backward()
            optim_a.step()
            scheduler_a.step()
        loss = model(images, targets)
        meters.update(loss=loss.item())
        optim_w.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optim_w.step()
        scheduler_w.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 50 == 0:
            logger.info(
                meters.delimiter.join(
                    ["eta: {eta}",
                     "iter: {epoch}/{iter}",
                     "{meters}",
                     "lr: {lr:.6f}"]).format(
                    eta=eta_string,
                    epoch=epoch,
                    iter=iteration,
                    meters=str(meters),
                    lr=optim_w.param_groups[0]['lr']))


def search(cfgb, output_dir):
    train_loaders, val_loaders = build_dataset(cfgb)
    model = build_model(cfgb)
    model = model.cuda()
    optimizer = make_optimizer(cfgb, model)
    scheduler = make_lr_scheduler(cfgb, optimizer, len(train_loaders[0]))
    checkpointer = Checkpointer(model, optimizer, scheduler, output_dir + '/models', save_to_disk=True)
    extra_checkpoint_data = checkpointer.load(cfgb.MODEL.WEIGHT)
    arguments = {
        "epoch": 0,
        **extra_checkpoint_data
    }
    checkpoint_period = cfgb.SOLVER.CHECKPOINT_PERIOD
    val_period = cfgb.SOLVER.VALIDATE_PERIOD
    max_epoch = cfgb.SOLVER.MAX_EPOCH
    arch_start_epoch = cfgb.SEARCH.ARCH_START_EPOCH
    writer = SummaryWriter(logdir=output_dir + '/log', comment=cfgb.DATASET.DATA_SET)
    do_search(
        model,
        train_loaders,
        val_loaders,
        max_epoch,
        arch_start_epoch,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpoint_period,
        arguments,
        writer,
        cfg,
        visual_dir=output_dir
    )


def main():
    parser = argparse.ArgumentParser(description="neural architecture search for water body identification")
    parser.add_argument("--config-file", default='./configs/gd/search.yaml', metavar="FILE",
                        help="path to config file", type=str)
    parser.add_argument("--device", default='0', help="path to config file", type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    output_dir = os.path.join(cfg.OUTPUT_DIR, '{}'.format(cfg.DATASET.DATA_SET), 'search')
    mkdir(output_dir)
    mkdir(os.path.join(output_dir, 'models'))
    logger = setup_logger("nas", output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    search(cfg, output_dir)


if __name__ == "__main__":
    main()
