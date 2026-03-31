import warnings

warnings.filterwarnings("ignore")
import os
import warnings
from sklearn import metrics
import  yaml


import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")

import time
import sys
import os

with open('/home/wxp/25week/MCFNet/efficientnet.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
    model_cfg = cfg['MODEL']
# import logging
import warnings
import numpy
import torch
import torch.nn as nn
from models.MCFNet import MCFNet

from datasets.dataset import DeepfakeDataset

import cv2
# from utils2 import dist_average, ACC, compute_metrics, compute_metrics2
from config0 import train_config

# from datasets.cli_utils import get_params
assert torch.cuda.is_available()

# args = get_params()

from torch.utils.data import RandomSampler, SubsetRandomSampler
import random

def main_worker_23():
    local_rank =4
    rank =3
    torch.manual_seed(49)
    np.random.seed(49)
    random.seed(49)
    # 若使用CUDA
    torch.cuda.manual_seed_all(49)  # y  tensor([1, 0, 0, 0, 1, 1, 1, 1, 1, 1], device='cuda:3')
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(local_rank)
    Config1 = train_config('ff-all-c23', batch_size=16, resize=(384, 384))
    config=Config1
    train_dataset = DeepfakeDataset(phase='train', **Config1.dataset)
    test_dataset1 = DeepfakeDataset(phase='test', **Config1.dataset)

    Config6 = train_config('celebv1', batch_size=20, resize=(384, 384))
    test_dataset6 = DeepfakeDataset(phase='test', **Config6.dataset)

    Config7 = train_config('dfd', batch_size=20, resize=(384, 384))
    test_dataset7 = DeepfakeDataset(phase='test', **Config7.dataset)

    Config8 = train_config('faceshifter', batch_size=20, resize=(384, 384))
    test_dataset8 = DeepfakeDataset(phase='test', **Config8.dataset)
    #
    Config9 = train_config('uadfv', batch_size=20, resize=(384, 384))
    test_dataset9 = DeepfakeDataset(phase='test', **Config9.dataset)

    Config2 = train_config('celebv2', batch_size=20, resize=(384, 384))
    test_dataset2 = DeepfakeDataset(phase='test', **Config2.dataset)

    Config3 = train_config('dfdc', batch_size=20, resize=(384, 384))
    test_dataset3 = DeepfakeDataset(phase='test', **Config3.dataset)

    Config4 = train_config('dfw', batch_size=20, resize=(384, 384))
    test_dataset4 = DeepfakeDataset(phase='test', **Config4.dataset)

    Config5 = train_config('dfdcp', batch_size=20, resize=(384, 384))
    test_dataset5 = DeepfakeDataset(phase='test', **Config5.dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config1.batch_size,
                                               sampler=RandomSampler(train_dataset),
                                               pin_memory=True, num_workers=Config1.workers)
    #
    test_loader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=Config1.batch_size,
                                               # sampler= RandomSampler(test_dataset1),
                                               pin_memory=True, num_workers=Config1.workers)
    #

    test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=Config1.batch_size,
                                               # sampler= RandomSampler(test_dataset2),
                                               pin_memory=True, num_workers=Config1.workers)

    test_loader3 = torch.utils.data.DataLoader(test_dataset3, batch_size=Config1.batch_size,
                                               # sampler= RandomSampler(test_dataset3),
                                               pin_memory=True, num_workers=Config1.workers)

    test_loader4 = torch.utils.data.DataLoader(test_dataset4, batch_size=Config1.batch_size,
                                               # sampler=RandomSampler(test_dataset4),
                                               pin_memory=True, num_workers=Config1.workers)
    test_loader5 = torch.utils.data.DataLoader(test_dataset5, batch_size=Config1.batch_size,

                                               # sampler= RandomSampler(test_dataset4),
                                               pin_memory=True, num_workers=Config1.workers)
    test_loader6 = torch.utils.data.DataLoader(test_dataset6, batch_size=Config1.batch_size,
                                               # sampler= RandomSampler(test_dataset2),
                                               pin_memory=True, num_workers=Config1.workers)

    test_loader7 = torch.utils.data.DataLoader(test_dataset7, batch_size=Config1.batch_size,
                                               # sampler= RandomSampler(test_dataset3),
                                               pin_memory=True, num_workers=Config1.workers)

    test_loader8 = torch.utils.data.DataLoader(test_dataset8, batch_size=Config1.batch_size,
                                               # sampler=RandomSampler(test_dataset4),
                                               pin_memory=True, num_workers=Config1.workers)
    test_loader9 = torch.utils.data.DataLoader(test_dataset9, batch_size=Config1.batch_size,
                                               # sampler= RandomSampler(test_dataset4),
                                               pin_memory=True, num_workers=Config1.workers)



    net = MCFNet(model_cfg)




    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(local_rank)
    # from AGDA import AGDA
    # AG = AGDA(**config.AGDA_config).to(local_rank)
    AG = None
    config.learning_rate = config.learning_rate * (config.batch_size / 80)
    config.learning_rate = 0.00025
    print(f"*config.learning_rate{config.learning_rate}  ")
    # handles = register_nan_hooks_to_submat(net.sub)

    # AG = AGDA(**config.AGDA_config).to(local_rank)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, betas=config.adam_betas,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step,
                                                gamma=config.scheduler_gamma)
    torch.cuda.empty_cache()
    start_epoch = 0
    val_list = []
    train_list = []
    val_auc = 0
    best_auc = 0
    val_acc = 0
    best_acc = 0
    net, optimizer, scheduler, start_epoch, best_acc, val_real_acc, val_fake_acc = load_checkpoint(
        net, optimizer, scheduler,
        '/mnt/data/wxp/25week/2021multiple-attention-master/对比算法/MINETEST/model_best.pth.tar'
    )
    print(f"Resuming from epoch {start_epoch} with best accuracy {best_acc}")
    print(f"Validation real acc: {val_real_acc}, fake acc: {val_fake_acc}")
    epoch = start_epoch
    print(f"**************************    Epoch {epoch} validating  in celebv1   **************************")

    val_loss_value, val_acc, val_real_acc, val_fake_acc, val_auc = run(epoch, data_loader=test_loader6, net=net,
                                                                       optimizer=optimizer, local_rank=local_rank,
                                                                       config=Config2, phase='val', AG=None)
    print(f"**************************    Epoch {epoch} validating  in celebv2   **************************")

    val_loss_value, val_acc, val_real_acc, val_fake_acc, val_auc = run(epoch, data_loader=test_loader2, net=net,
                                                                       optimizer=optimizer, local_rank=local_rank,
                                                                       config=Config2, phase='val', AG=None)
    print(f"**************************    Epoch {epoch} validating  in dfdc   **************************")
    val_loss_value, val_acc, val_real_acc, val_fake_acc, val_auc = run(epoch, data_loader=test_loader3, net=net,
                                                                       optimizer=optimizer, local_rank=local_rank,
                                                                       config=Config3, phase='val', AG=None)
    # [Metrics]# ACC = 0.4753 | F1 = 0.4578 | AUC = 0.4670 | Real_ACC = 0.4252(68851) | Fake_ACC = 0.5298(63265
    print(f"**************************    Epoch {epoch} validating  in dfw   **************************")
    val_loss_value, val_acc, val_real_acc, val_fake_acc, val_auc = run(epoch, data_loader=test_loader4, net=net,
                                                                       optimizer=optimizer, local_rank=local_rank,
                                                                       config=Config4, phase='val', AG=None)


import torch.nn.functional as F

import numpy as np
import logging
from tqdm import tqdm


def run(epoch, data_loader, net, optimizer, local_rank, config, phase, AG):
    if phase == 'train':
        net.train()
    else:
        net.eval()

    data_length, acc, val_loss = 0, 0, 0
    temp_total_loss, temp_total_acc = 0, 0
    data_loader_length = len(data_loader)

    # total_step = data_loader_length / getattr(args, "batch_size")
    # 仅用于test
    out_list, label_list = [], []

    for i, datas in enumerate(data_loader):
        # data_loader_train = tqdm(data_loader, desc='Train', ncols=60)

        images, labels,_ = datas
        data_length += len(images)
        X = images.to(local_rank, non_blocking=True)
        y = labels.to(local_rank, non_blocking=True)
        # try:
        #     # 一次正常的 forward
        #     out = net(images, labels)
        # except RuntimeError as e:
        #     print("发现 NaN/Inf 的具体位置：", e)
        if phase == 'train':
            # from torch import autocast
            # torch.autograd.set_detect_anomaly(True)
            # with autocast(device_type='cuda', dtype=torch.float16):
            with torch.set_grad_enabled(phase == 'train'):
                # logits ,output1= net(X )
                # rec, ah,logits= net(X )
                logits= net(X )

        else:
            # from torch import autocast
            # with autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                # logits ,output1= net(X )
                # rec, ah,logits= net(X )
                logits= net(X )
                # y: [B] (0 for real, 1 for fake)
        # real_mask = (y == 0).float().view(-1, 1, 1, 1)  # [B,1,1,1]，用于选择real样本
        # # print(f"type(outputs): {type(outputs)}")
        # # inspect_outputs(x)
        #
        # # 仅保留 real 样本的预测与原图
        # real_output = rec * real_mask
        # real_input = X * real_mask
        #
        # batch_loss =F.cross_entropy(logits, y)+ 0.1*total_loss(rec, X) #+ total_loss(output2, mask) / 4 + total_loss(output3, mask) / 8)
        # batch_loss =F.cross_entropy(logits, y)+ 0.1*total_loss(real_output, real_input) #+ total_loss(output2, mask) / 4 + total_loss(output3, mask) / 8)
        # print(F.cross_entropy(logits, y),total_loss(output1, X))#ensor(0.6323, device='cuda:4', grad_fn=<NllLossBackward0>) tensor(0.4217, device='cuda:4', grad_fn=<AddBackward0>)
        # batch_loss = train_loss(loss_pack, config)
        batch_loss = F.cross_entropy(logits, y)
        batch_loss_val = batch_loss.detach()
        if batch_loss_val.dim() > 0:
            batch_loss_val = batch_loss_val.mean()  # 或者 .sum() 根据你的策略
        temp_total_loss += batch_loss_val.item()
        # temp_total_loss += float(batch_loss.detach().cpu().numpy())
        # print(f"batch_loss: {batch_loss.detach().cpu().numpy()}")
        if phase == 'train':
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # acc, batch_real_acc, batch_fake_acc, batch_real_cnt, batch_fake_cnt,auc = compute_metrics2(logits, y)
        label_list.extend(y)
        out_list.extend(logits)
        # if i % 1000 == 0:
        #     print(f"val_auc{round(float(auc), 4)}, "
        #           f"acc_{round(float(acc), 4)}, "
        #            f"real_acc_{round(float(batch_real_acc), 4) if batch_real_cnt != 0 else 'NAN'}, "
        #            f"fake_acc_{round(float(batch_fake_acc), 4) if batch_fake_cnt != 0 else 'NAN'}, "
        #            f"loss_{round(float(batch_loss.detach().cpu().numpy()), 5)}")
    outs = torch.stack(out_list)
    outs = outs.reshape(-1, 2)
    ys = torch.stack(label_list)
    ys = ys.reshape(-1)
    val_acc, val_f1, val_auc, val_real_acc, val_fake_acc, real_cnt, fake_cnt = compute_metrics_all(outs, ys)

    return round(temp_total_loss / data_length, 5), round(float(val_acc), 4), round(float(val_real_acc), 4), round(float(val_fake_acc), 4), round(float(val_auc), 4)


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, val_real_acc, val_fake_acc, val_auc, save_model_path,
                    suffix, filename='checkpoint.pth.tar'):
    os.makedirs(save_model_path, exist_ok=True)
    filename_best = 'model_best.pth.tar'
    filename_best = os.path.join(save_model_path, filename_best)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'sched_state': scheduler.state_dict(),
        'best_acc': best_acc,
        'val_real_acc': val_real_acc,
        'val_fake_acc': val_fake_acc,
        'auc': val_auc
    }, filename_best)
    # torch.save(model.state_dict(), filename_best)
    # shutil.copyfile(filename, filename_best)
    print(f"[INFO] Saved checkpoint to {filename_best}")

    logging.info('save model {filename}'.format(filename=filename_best))
    info_path = os.path.join(save_model_path, 'best_model_info.txt')
    with open(info_path, 'w') as f:
        f.write(f'Epoch: {suffix}\n')
        f.write(f'Best auc: {auc:.4f}\n')
        f.write(f'Val Real Acc: {val_real_acc:.4f}\n')
        f.write(f'Val Fake Acc: {val_fake_acc:.4f}\n')
    return filename_best


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    print("模型加载成功（已忽略不匹配的参数）")
    # optimizer.load_state_dict(checkpoint['optim_state'])
    # scheduler.load_state_dict(checkpoint['sched_state'])

    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    val_real_acc = checkpoint['val_real_acc']
    val_fake_acc = checkpoint['val_fake_acc']

    return model, optimizer, scheduler, epoch, best_acc, val_real_acc, val_fake_acc


def distributed_train(config, world_size=0, num_gpus=0, rank_offset=0):
    if not num_gpus:
        num_gpus = torch.cuda.device_count()
    if not world_size:
        world_size = num_gpus
    # mp.spawn(main_worker, nprocs=num_gpus, args=(world_size, rank_offset, config))
    main_worker(config)
    torch.cuda.empty_cache()

def inspect_outputs(outputs, prefix="outputs"):
    """
    自动递归查看模型输出（支持 tensor / tuple / list / dict）
    """
    if torch.is_tensor(outputs):
        print(f"{prefix}: Tensor shape={tuple(outputs.shape)} dtype={outputs.dtype}")
    elif isinstance(outputs, (list, tuple)):
        print(f"{prefix}: {type(outputs).__name__} (len={len(outputs)})")
        for i, item in enumerate(outputs):
            inspect_outputs(item, prefix=f"{prefix}[{i}]")
    elif isinstance(outputs, dict):
        print(f"{prefix}: dict (len={len(outputs)})")
        for k, v in outputs.items():
            inspect_outputs(v, prefix=f"{prefix}['{k}']")
    else:
        print(f"{prefix}: {type(outputs)} (non-tensor value)")

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def compute_metrics_all(outs, ys, threshold=0.5, verbose=True):
    """
    计算分类指标，支持二分类和五分类
    返回统一格式: acc, f1, auc, real_acc, fake_acc, real_cnt, fake_cnt
    对于五分类: real_acc=类别1的准确率, fake_acc=类别0的准确率
    """
    # 自动判断任务类型
    if outs.ndim == 1:
        task_type = "binary"
        probs = torch.sigmoid(outs).detach().cpu().numpy()
        preds = (probs >= threshold).astype(int)
    else:
        if outs.shape[1] == 2:
            task_type = "binary"
            probs = torch.softmax(outs, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(outs, dim=1).detach().cpu().numpy()
        elif outs.shape[1] == 5:
            task_type = "multiclass"
            probs = torch.softmax(outs, dim=1).detach().cpu().numpy()
            preds = torch.argmax(outs, dim=1).detach().cpu().numpy()
        else:
            raise ValueError(f"不支持的输出维度: {outs.shape}")

    ys = ys.detach().cpu().numpy()

    if task_type == "binary":
        # 二分类逻辑（与原函数相同）
        acc = accuracy_score(ys, preds)
        f1 = f1_score(ys, preds, zero_division=0)
        try:
            auc = roc_auc_score(ys, probs)
        except ValueError:
            auc = float("nan")

        cm = confusion_matrix(ys, preds, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()

        real_cnt = TP + FN
        real_acc = TP / real_cnt if real_cnt > 0 else 0.0
        fake_cnt = TN + FP
        fake_acc = TN / fake_cnt if fake_cnt > 0 else 0.0

        if verbose:
            print(f"[Binary] ACC={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | "
                  f"Real_ACC={real_acc:.4f} | Fake_ACC={fake_acc:.4f}")

        return acc, f1, auc, real_acc, fake_acc, real_cnt, fake_cnt

    else:
        # 五分类逻辑
        acc = accuracy_score(ys, preds)
        f1 = f1_score(ys, preds, average='macro', zero_division=0)

        try:
            auc = roc_auc_score(ys, probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = float("nan")

        # 计算所有五个类别的准确率
        cm = confusion_matrix(ys, preds, labels=[0, 1, 2, 3, 4])
        class_accuracies = []
        class_counts = []

        for i in range(5):
            correct = cm[i, i]  # 对角线元素，正确分类的数量
            total = cm[i, :].sum()  # 该类别总样本数
            class_acc = correct / total if total > 0 else 0.0
            class_accuracies.append(class_acc)
            class_counts.append(total)

        # 为了保持返回格式兼容，仍然使用real_acc和fake_acc
        fake_acc = class_accuracies[0]  # 类别0的准确率
        real_acc = class_accuracies[1]  # 类别1的准确率
        fake_cnt = class_counts[0]  # 类别0的样本数
        real_cnt = class_counts[1]  # 类别1的样本数

        if verbose:
            print(f"[5-Class] ACC={acc:.4f} | F1_macro={f1:.4f} | AUC={auc:.4f}")
            for i in range(5):
                print(f"                Class_{i}_ACC={class_accuracies[i]:.4f} ({class_counts[i]} samples)")

        return acc, f1, auc, real_acc, fake_acc, real_cnt, fake_cnt


if __name__ == '__main__':
    from config0 import train_config
    import linecache
    import sys
    import torch
    import sys
    import codecs
    import linecache

    torch.cuda.empty_cache()
    main_worker_23()
