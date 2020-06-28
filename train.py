import os
import gc
import time
import datetime
import random
import warnings

warnings.simplefilter("ignore")

import numpy as np
import cv2
import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm.notebook import tqdm

# from sklearn.metrics import roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.serialization as xser
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# from ranger import Ranger
from catalyst.data.sampler import DistributedSamplerWrapper, BalanceClassSampler

from augmentations import get_train_transforms, get_valid_transforms
from loss import sigmoid_focal_loss, bce_criterion
from metric import RocAucMeter
from models import EfficientNet
from dataset import MelanomaDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    xm.set_rng_state(seed, device=xm.xla_device())


DATA_DIR = "/content/data/"

df_train = pd.read_csv(DATA_DIR + "folds_13062020.csv")
df_test = pd.read_csv(DATA_DIR + "test.csv").rename(columns={"image_name": "image_id"})
sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")


def train_model(data, fold_no, log=False):
    seed_everything(FLAGS["seed"])

    def get_datasets(data):
        X_train, y_train, X_val, y_val = data
        datasets = {}
        datasets["train"] = MelanomaDataset(
            X_train, y_train, istrain=True, transforms=get_train_transforms()
        )
        datasets["valid"] = MelanomaDataset(
            X_val, y_val, istrain=False, transforms=get_valid_transforms()
        )
        return datasets

    datasets = SERIAL_EXEC.run(lambda: get_datasets(data))

    if xm.is_master_ordinal == True and log == True:
        writer = SummaryWriter()
        # writer.add_hparams(FLAGS)

    labels_vcount = y_train["target"].value_counts()
    class_counts = [
        labels_vcount[0].astype(np.float32),
        labels_vcount[1].astype(np.float32),
    ]
    num_samples = sum(class_counts)
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    weights = [
        class_weights[y_train["target"].values[i]] for i in range(int(num_samples))
    ]
    wrsampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    # BalanceClassSampler(labels=y_train['target'].values, mode="downsampling"),

    # DistributedSamplerWrapper
    train_sampler = DistributedSamplerWrapper(
        sampler=wrsampler,  # sampler=wrsampler,# datasets['train'],
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
    )
    validation_sampler = DistributedSampler(
        datasets["valid"],
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
    )
    train_loader = DataLoader(
        datasets["train"],
        batch_size=FLAGS["batch_size"],
        num_workers=FLAGS["num_workers"],
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        datasets["valid"],
        batch_size=FLAGS["batch_size"],
        num_workers=FLAGS["num_workers"],
        sampler=validation_sampler,
        drop_last=True,
    )

    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FLAGS["learning_rate"] * xm.xrt_world_size(),
        weight_decay=FLAGS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        cooldown=1,
        mode="min",
        patience=2,
        verbose=True,
        min_lr=1e-8,
    )

    criterion = sigmoid_focal_loss

    def train_one_epoch(loader):
        model.train()
        running_loss = 0
        max_idx = 0
        xm.master_print("-" * 40)
        xm.master_print("Step\t|\tTime")
        xm.master_print("-" * 40)
        for idx, (images, targets) in enumerate(loader):
            optimizer.zero_grad()
            y_pred = model(images.float())
            loss = criterion(y_pred, targets)
            running_loss += float(loss)
            loss.backward()
            xm.optimizer_step(optimizer)
            # xm.mark_step() call everystep for grad accum
            max_idx = float(idx)
            if idx % FLAGS["log_steps"] == 0 and idx != 0:
                xm.master_print(
                    "({})\t|\t{}".format(idx, time.asctime(time.localtime()))
                )
        xm.master_print("-" * 40)
        return running_loss / (max_idx + 1)

    def val_one_epoch(loader):
        model.eval()
        running_loss = 0
        max_idx = 0
        roc_auc_scores = RocAucMeter()
        with torch.no_grad():
            for idx, (images, targets) in enumerate(loader):
                y_pred = model(images.float())
                loss = criterion(y_pred, targets)
                running_loss += float(loss)
                max_idx = float(idx)
                roc_auc_scores.update(targets, y_pred)  # [:, 1]
        score = roc_auc_scores.avg
        return running_loss / (max_idx + 1), score

    def _reduce_fn(x):
        return np.array(x).mean()

    best_score = 0
    xm.master_print("=" * 26 + f"Fold #{fold_no} started" + "=" * 27)
    for epoch in range(0, FLAGS["num_epochs"]):
        xm.master_print("-" * 26 + f"Epoch #{epoch+1} started" + "-" * 26)
        xm.master_print(f"Epoch start {time.asctime(time.localtime())}")
        train_start = time.time()
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loss = train_one_epoch(para_loader.per_device_loader(device))
        xm.master_print(f"finished training epoch {epoch+1}")
        elapsed_time = int(time.time() - train_start)
        xm.master_print(f"elapsed time: {(elapsed_time)//60}mins {(elapsed_time)%60}s")
        reduced_loss = xm.mesh_reduce("train_loss", train_loss, _reduce_fn)
        xm.master_print(f"reduced loss {reduced_loss:.5f}")
        if xm.is_master_ordinal == True and log == True:
            writer.add_scalar("train/loss", reduced_loss, epoch + 1)

        if (epoch + 1) % FLAGS["val_freq"] == 0:
            val_start = time.time()
            para_loader = pl.ParallelLoader(val_loader, [device])
            val_loss, auc_score = val_one_epoch(para_loader.per_device_loader(device))
            xm.master_print(f"finished validating epoch {epoch+1}")
            reduced_val_loss = xm.mesh_reduce("val_loss", val_loss, _reduce_fn)
            reduced_auc_score = xm.mesh_reduce("auc_score", auc_score, _reduce_fn)
            scheduler.step(reduced_val_loss)
            xm.master_print(f"reduced val loss {reduced_val_loss:.5f}")
            xm.master_print(f"reduced auc score {reduced_auc_score:.5f}")
            val_elapsed_time = int(time.time() - val_start)
            xm.master_print(
                f"elapsed time: {(val_elapsed_time)//60}mins {(val_elapsed_time)%60}s"
            )
            if xm.is_master_ordinal == True and log == True:
                writer.add_scalar("val/loss", reduced_val_loss, epoch + 1)
                writer.add_scalar("val/roc_auc", reduced_auc_score, epoch + 1)
            if (
                best_score < reduced_auc_score
                or (best_score - reduced_auc_score) < 0.005
            ):
                best_score = reduced_auc_score
                file_name = f"./{FLAGS['exp_name']}_fold_{fold_no+1}_epoch_{epoch+1}_auc_{reduced_auc_score:.5f}.pth"
                xm.save(model.state_dict(), file_name)
                xm.master_print(f"saved model...")
                xm.master_print(f"new best score: {best_score:.5f}")
                # xser.save(model.state_dict(), file_name, master_only=True)

        xm.master_print(f"Epoch end {time.asctime(time.localtime())}")
        xm.master_print("-" * 27 + f"Epoch #{epoch+1} ended" + "-" * 26)

    xm.master_print("=" * 28 + f"Fold #{fold_no} ended" + "=" * 27)


def _mp_fn(rank, flags, data, fold_no, log):
    global FLAGS
    global WRAPPED_MODEL
    global SERIAL_EXEC
    FLAGS = flags
    torch.set_default_tensor_type("torch.FloatTensor")
    train_model(data, fold_no, log)


FLAGS = {}
FLAGS["batch_size"] = 64
FLAGS["num_workers"] = 8
FLAGS["learning_rate"] = 4e-4
FLAGS["num_epochs"] = 40
FLAGS["weight_decay"] = 1e-4
FLAGS["log_steps"] = 20
FLAGS["img_size"] = 300
FLAGS["loss"] = "focal"
FLAGS["optimizer"] = "AdamW"
FLAGS["scheduler"] = "ReduceLROnPlateau"
FLAGS["exp_name"] = "enet_b0"
FLAGS["fold"] = [0]  # , 1, 2, 3, 4]
FLAGS["val_freq"] = 1
FLAGS["num_cores"] = 8
FLAGS["seed"] = 42

model_cpu = EfficientNet("tf_efficientnet_b0_ns")
WRAPPED_MODEL = xmp.MpModelWrapper(model_cpu)
SERIAL_EXEC = xmp.MpSerialExecutor()

for fold_no in FLAGS["fold"]:
    X_train = df_train[df_train["fold"] != fold_no][
        [col for col in df_train.columns if col != "target"]
    ]
    X_val = df_train[df_train["fold"] == fold_no][
        [col for col in df_train.columns if col != "target"]
    ]
    y_train = df_train[df_train["fold"] != fold_no][
        [col for col in df_train.columns if col == "target"]
    ]
    y_val = df_train[df_train["fold"] == fold_no][
        [col for col in df_train.columns if col == "target"]
    ]
    data = X_train, y_train, X_val, y_val
    xmp.spawn(
        _mp_fn,
        args=(FLAGS, data, fold_no, False),
        nprocs=FLAGS["num_cores"],
        start_method="fork",
    )

