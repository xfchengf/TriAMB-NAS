import argparse
import os
import logging
import time
import datetime
import PIL.Image as Image
import h5py
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import errno
import sys
from nas import cfg
from nas import build_dataset
from nas import make_lr_scheduler
from nas import make_optimizer
from nas import Checkpointer
from nas import MetricLogger
from tensorboardX import SummaryWriter
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score, jaccard_score
from nas import color_dict
from nas import HSIdatasettest
from nas import HSItrainnet
from nas import HSIsearchnet
from nas import model_visualize
from nas import OptimizerDict

ARCHITECTURES = {
    "searchnet": HSIsearchnet,
    "trainnet": HSItrainnet,
}


def compute_params(model):
    n_params = 0
    for m in model.parameters():
        n_params += m.numel()
    return n_params


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
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            pred = model(images)  # 混合精度
            result_anal(pred, targets)
        acc, loss = result_anal.get_result()
        result_anal.reset()
    return acc, loss


def h5_data_loader(data_dir):
    with h5py.File(data_dir, 'r') as g:
        data = g['data'][:]
        label = g['label'][:]
    return data, label


def h5_dist_loader2(data_dir):
    with h5py.File(data_dir, 'r') as h:
        height, width = h['height'][0], h['width'][0]
        category_num = h['category_num'][0]
        test_map = h['test_label_map'][0]
    return height, width, category_num, test_map


def get_patches_list2(height, width, crop_size, overlap):
    patch_list = []
    if overlap:
        slide_step = crop_size // 2
    else:
        slide_step = crop_size
    x1_list = list(range(0, width - crop_size, slide_step))
    y1_list = list(range(0, height - crop_size, slide_step))
    x1_list.append(width - crop_size)
    y1_list.append(height - crop_size)
    x2_list = [x + crop_size for x in x1_list]
    y2_list = [y + crop_size for y in y1_list]
    for x1, x2 in zip(x1_list, x2_list):
        for y1, y2 in zip(y1_list, y2_list):
            patch = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
            patch_list.append(patch)
    return patch_list


def oa_aa_k_cal(pre_label, tar_label):
    test_indices = np.where(tar_label > 0)
    pre_label = pre_label[test_indices]
    tar_label = tar_label[test_indices]
    acc = []
    samples_num = len(tar_label)
    category_num = tar_label.max()
    for i in range(1, int(category_num) + 1):
        loc_i = np.where(tar_label == i)
        oa_i = np.array(pre_label[loc_i] == tar_label[loc_i], np.float32).sum() / len(loc_i[0])
        acc.append(oa_i)
    oa = np.array(pre_label == tar_label, np.float32).sum() / samples_num
    aa = np.average(np.array(acc))
    k = cohen_kappa_score(tar_label, pre_label)
    return oa, aa, k, np.array(acc)


def labelmap_2_img(color_list, label_map):
    h, w = label_map.shape
    img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            r, g, b = color_list[str(label_map[i, j])]
            img[i, j] = [r, g, b]
    return np.array(img, np.uint8)


def do_train(
        model,
        train_loader,
        val_loader,
        max_iter,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        arguments,
        writer):
    logger = logging.getLogger("nas.trainer")
    logger.info(f"Model Params: {compute_params(model) / 1000:.2f}K")
    logger.info("Start training")
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    train_anal = ResultAnalysis()
    val_best_acc, val_min_loss = 0, 10
    model.train()
    data_iter = iter(train_loader)
    meters = MetricLogger(delimiter="  ")
    end = time.time()
    for iteration in range(start_iter, max_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        # 检查模型是否在 GPU 上，如果是，将输入数据也移动到 GPU 上
        if next(model.parameters()).is_cuda:
            images = images.cuda()
            targets = targets.cuda()
        data_time = time.time() - end
        if isinstance(model, HSIsearchnet) and cfg.SEARCH.SEARCH_ON:
            loss = model(images, targets)
        else:
            pred, loss = model(images, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
        if isinstance(optimizer, OptimizerDict):
            optimizer['optim_w'].step()
            optimizer['optim_a'].step()
            scheduler['scheduler_w'].step()
            scheduler['scheduler_a'].step()
        else:
            optimizer.step()
            scheduler.step()
        if not isinstance(model, HSIsearchnet) or not cfg.SEARCH.SEARCH_ON:
            train_anal(pred, targets)
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % (val_period // 4) == 0:
            logger.info(
                meters.delimiter.join(
                    ["eta: {eta}",
                     "iter: {iter}",
                     "{meters}",
                     "lr: {lr:.6f}",
                     "max_mem: {memory:.0f}"]).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
        if iteration % val_period == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
            train_acc, train_loss = train_anal.get_result()
            train_anal.reset()
            if iteration % val_period == 0:
                val_acc, val_loss = inference(model, val_loader)
                print("val_acc:{} val_loss:{}".format(val_acc, val_loss))
                if val_acc > val_best_acc:
                    val_best_acc = val_acc
                    val_min_loss = val_loss
                    checkpointer.save("model_best", **arguments)
                elif val_acc == val_best_acc and val_loss < val_min_loss:
                    val_best_acc = val_acc
                    val_min_loss = val_loss
                    checkpointer.save("model_best", **arguments)
                model.train()
                writer.add_scalars('overall_acc', {'train_acc': train_acc, 'val_acc': val_best_acc}, iteration)
                writer.add_scalars('loss', {'train_loss': train_loss, 'val_psnr': val_min_loss}, iteration)
            else:
                writer.add_scalars('overall_acc', {'train_acc': train_acc}, iteration)
                writer.add_scalars('loss', {'train_loss': train_loss}, iteration)
        if iteration % val_period == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(f"Total training time: {total_time_str}")
    writer.close()


def train2(cfgb, output_dir):
    model = build_model(cfgb)
    model = model.cuda()  # 单卡环境#
    geno_file = os.path.join(cfgb.OUTPUT_DIR, '{}'.format(cfgb.DATASET.DATA_SET), 'search/models/model_best.geno')
    genotype = torch.load(geno_file, map_location=torch.device("cpu"), weights_only=True)
    gene_cell = genotype
    visual_dir = output_dir
    best_model_visualization_dir = os.path.join(visual_dir, 'visualize', 'best_model')
    model_visualize(best_model_visualization_dir)
    if cfgb.SEARCH.SEARCH_ON:
        optimizer = make_optimizer(cfgb, model)
    else:
        optimizer = make_optimizer(cfgb, model)
    scheduler = make_lr_scheduler(cfgb, optimizer)
    checkpointer = Checkpointer(model, optimizer, scheduler, os.path.join(output_dir, 'models'), save_to_disk=True)
    train_loader, val_loader = build_dataset(cfgb)
    extra_checkpoint_data = checkpointer.load(cfgb.MODEL.WEIGHT)
    arguments = {
        "iteration": 0,
        "gene_cell": gene_cell,
        **extra_checkpoint_data
    }
    arguments.update(extra_checkpoint_data)
    val_period = cfgb.SOLVER.VALIDATE_PERIOD
    max_iter = cfgb.SOLVER.TRAIN.MAX_ITER
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'), comment=cfgb.DATASET.DATA_SET)
    do_train(
        model,
        train_loader,
        val_loader,
        max_iter,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        arguments,
        writer)


def evaluation(cfgc):
    print('model build')
    trained_model_dir = os.path.join(cfgc.OUTPUT_DIR, '{}'.format(cfgc.DATASET.DATA_SET),
                                     'train{}'.format(cfgc.DATASET.TRAIN_NUM), 'models/model_best.pth')
    if not os.path.exists(trained_model_dir):
        print('trained_model does not exist')
        return None, None
    model = build_model(cfgc)
    model = model.cuda()  # 单卡环境
    model_state_dict = torch.load(trained_model_dir, weights_only=True).pop("model")
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError:
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
    model.eval()
    print('load test set')
    data_root = cfgc.DATASET.DATA_ROOT
    data_set = cfgc.DATASET.DATA_SET
    batch_size = cfgc.DATALOADER.BATCH_SIZE_TEST
    dataset_dir = os.path.join(data_root, f'{data_set}.h5')
    dataset_dist_dir = os.path.join(cfgc.DATALOADER.DATA_LIST_DIR, '{}_dist_{}_train-{}_val-{}.h5'
                                    .format(data_set, cfgc.DATASET.DIST_MODE, float(cfgc.DATASET.TRAIN_NUM),
                                            float(cfgc.DATASET.VAL_NUM)))
    test_data, label_map = h5_data_loader(dataset_dir)
    height, width, category_num, test_map = h5_dist_loader2(dataset_dist_dir)
    result_save_dir = os.path.join(cfgc.OUTPUT_DIR, '{}'.format(cfgc.DATASET.DATA_SET),
                                   'eval_{}'.format(cfgc.DATASET.TRAIN_NUM))
    mkdir(result_save_dir)
    crop_size = cfgc.DATASET.CROP_SIZE
    overlap = cfgc.DATASET.OVERLAP
    test_patches_list = get_patches_list2(height, width, crop_size, overlap)
    dataset_test = HSIdatasettest(hsi_data=test_data, data_dict=test_patches_list)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=batch_size, pin_memory=True)
    print('dataset {} evaluation...'.format(data_set))
    pred_score_map = torch.zeros(category_num, height, width)
    pred_count_map = torch.zeros(height, width)
    time_sum = 0
    count = 0
    with torch.no_grad():
        for patches, indieces in test_loader:
            patches = patches.cuda()
            count += 1
            time_s = time.time()
            pred = model(patches)
            torch.cuda.synchronize()
            time_e = time.time()
            time_sum += (time_e - time_s)
            for i, [x1, x2, y1, y2] in enumerate(zip(indieces[0], indieces[1], indieces[2], indieces[3])):
                pred_patch = torch.softmax(pred[i].cpu(), dim=0)
                pred_score_map[:, y1:y2, x1:x2] += pred_patch
                pred_count_map[y1:y2, x1:x2] += 1
        pred_score_map = pred_score_map / pred_count_map.unsqueeze(dim=0)
    pred_map = torch.argmax(pred_score_map, dim=0) + 1
    print('time_cost:{} cout:{}'.format(time_sum / count, count))
    print('done')
    oa, aa, k, acc = oa_aa_k_cal(np.array(pred_map, np.float16), test_map)
    print('OA:{} AA:{} K:{}'.format(oa, aa, k))
    print(acc)
    with open(os.path.join(result_save_dir, 'evaluation_result.txt'), 'w') as f:
        f.write('OA: {}\n'.format(oa))
        f.write('AA: {}\n'.format(aa))
        f.write('K: {}\n'.format(k))
        for i in range(len(acc)):
            f.write('class {} acc: {}\n'.format(i + 1, acc[i]))
    if not cfgc.DATASET.SHOW_ALL:
        pred_map[np.where(label_map == 0)] = 0
    img_result = labelmap_2_img(color_dict[data_set], np.array(pred_map))
    img = Image.fromarray(img_result)
    img.save(os.path.join(result_save_dir, f'{data_set}.png'))


def main():
    parser = argparse.ArgumentParser(description="NAS Training and evaluation")
    parser.add_argument("--config-file", default='./configs/Houston2018/train.yaml', metavar="FILE",
                        help="path to config file", type=str)
    parser.add_argument("--device", default='0', help="path to config file", type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    output_dir = os.path.join(cfg.OUTPUT_DIR, '{}'.format(cfg.DATASET.DATA_SET),
                              'train{}'.format(cfg.DATASET.TRAIN_NUM))
    mkdir(output_dir + '/models')
    logger = setup_logger("nas", output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    train2(cfg, output_dir)
    evaluation(cfg)


if __name__ == "__main__":
    main()
