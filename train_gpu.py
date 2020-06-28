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
from tqdm import tqdm

# from sklearn.metrics import roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler

from torch.utils.data.distributed import DistributedSampler

# from ranger import Ranger
import neptune

from augmentations import get_train_transforms, get_valid_transforms
from loss import sigmoid_focal_loss, bce_criterion
from metric import RocAucMeter
from models import EfficientNet
from dataset import MelanomaDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_DIR = "/content/data/"

df_train = pd.read_csv(DATA_DIR + "folds_13062020.csv")
df_test = pd.read_csv(DATA_DIR + "test.csv").rename(columns={"image_name": "image_id"})
sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")


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


def train_one_epoch(
    loader, model, optimizer, epoch, scheduler=None, scaler=None, log=True
):
    model.train()
    running_loss = 0.0
    for idx, (images, targets) in tqdm(enumerate(loader), total=len(loader)):  #
        images = images.to(device)
        targets = targets.to(device)  # .unsqueeze(1)

        optimizer.zero_grad()

        y_pred = model(images.float())
        loss = rank_loss(y_pred, targets)
        # if np.random.rand()<0.33:
        #    if np.random.rand()<0.5:
        #        images, targets = cutmix(images, targets)
        #    else:
        #         images, targets = mixup(images, targets)
        #    y_pred = model(images.float())
        #    #if epoch<9:
        #    loss = cutmix_mixup_criterion(y_pred, targets, None, 0.05)
        #    #else:
        #    #    loss = cutmix_mixup_criterion(y_pred, targets, 0.9, 0.)
        # else:
        #    y_pred = model(images.float())
        #    #if epoch<9:
        #    #loss = smooth_ohem_criterion(y_pred, targets, 1.0, 0.)
        #    loss = smooth_criterion(y_pred, targets, 0.05)
        #    #else:
        #    #    loss = smooth_ohem_criterion(y_pred, targets, 0.9, 0.)

        running_loss += float(loss)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if log:
            neptune.log_metric("Loss/train", float(loss))

    return running_loss / len(loader)


def val_one_epoch(loader, model):
    model.eval()
    running_loss = 0.0
    y_preds_list = []
    targets_list = []
    roc_auc_scores = RocAucMeter()
    criterion = sigmoid_focal_loss
    with torch.no_grad():
        for idx, (images, targets) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device)
            targets = targets.to(device)  # .unsqueeze(1)
            y_pred = model(images.float())
            loss = criterion(y_pred, targets)
            running_loss += float(loss)
            roc_auc_scores.update(targets, y_pred)
    score = roc_auc_scores.avg
    # roc_plot = plot_roc(y_true, y_pred)
    print(f"roc_auc_score: {score}")
    print(f"average loss for val epoch: {running_loss/len(loader)}")
    return running_loss / len(loader), score  # , roc_plot


def save_upload(model, optimizer, best_score, epoch, fold=None, exp_name="model"):
    if fold:
        NAME = "siim-isic_" + exp_name + f"_fold_{str(fold+1)}_{str(epoch+1)}.ckpt"
    NAME = "siim-isic_" + exp_name + f"_{str(epoch+1)}.ckpt"
    MODEL_PATH = NAME
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        MODEL_PATH,
    )
    print(f"Saved ckpt for epoch {epoch+1}, new best score: {best_score}")
    upload_blob(MODEL_PATH, NAME)
    print(f"Uploaded ckpt for epoch {epoch+1}")


def fit(data, fold=None, log=True):
    exp_name = FLAGS["exp_name"]
    best_score = 0.0

    model = EfficientNet("tf_efficientnet_b0_ns").to(device)

    if log:
        neptune.init("utsav/SIIM-ISIC", api_token=NEPTUNE_API_TOKEN)
        neptune.create_experiment(
            exp_name, exp_description, params=FLAGS, upload_source_files="*.txt"
        )

    # optimizer = torch.optim.SGD(
    #    model.parameters(), lr=FLAGS['learning_rate'],
    #    momentum=FLAGS['momentum'],
    #    weight_decay=FLAGS['weight_decay']
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FLAGS["learning_rate"],
        weight_decay=FLAGS["weight_decay"],
    )

    # scheduler = ...
    # todo ...

    datasets = get_datasets(data)

    # sampler
    # labels_vcount = pd.Series(y_train).value_counts()
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
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    # loaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=FLAGS["batch_size"],
        num_workers=FLAGS["num_workers"],
        sampler=sampler,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["valid"],
        batch_size=32,
        shuffle=False,
        num_workers=FLAGS["num_workers"],
        drop_last=True,
    )

    scaler = GradScaler()
    # train loop
    for epoch in range(0, FLAGS["num_epochs"]):

        print("-" * 27 + f"Epoch #{epoch+1} started" + "-" * 27)

        train_loss = train_one_epoch(
            train_loader,
            model,
            optimizer,
            epoch,
            scheduler=None,
            scaler=scaler,
            log=log,
        )

        print()
        print(f"Average loss for epoch #{epoch+1} : {train_loss}")
        val_loss, auc_score = val_one_epoch(val_loader, model)

        # logs
        if log:
            neptune.log_metric("AUC/val", auc_score)
            # neptune.log_image("ROC/val", roc_plot)
            neptune.log_metric("Loss/val", val_loss)

        # checkpoint+upload
        if (auc_score > best_score) or (best_score - auc_score < 0.02):
            if auc_score > best_score:
                best_score = auc_score
            save_upload(model, optimizer, best_score, epoch, fold, exp_name=exp_name)

        print("-" * 28 + f"Epoch #{epoch+1} ended" + "-" * 28)
    if log:
        neptune.stop()
    return model


IMG_SIZE = 300
FLAGS = {}
FLAGS["batch_size"] = 32
FLAGS["num_workers"] = 4
FLAGS["learning_rate"] = 3e-4
FLAGS["num_epochs"] = 20
FLAGS["weight_decay"] = 5e-4
FLAGS["momentum"] = 0.9
FLAGS["img_size"] = IMG_SIZE
FLAGS["loss"] = "BCE unbalanced hard label ohem"
FLAGS["optimizer"] = "AdamW"
FLAGS["exp_name"] = "enet_b0"
FLAGS["fold"] = [0]  # , 1, 2, 3, 4]
exp_description = """
enet_b0 with base head,
Extra Data + Color fix
hard label ohem
cutmix + mixup
RandomWeightedSampler,
imsize 300
"""
log = False

try:
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
        trained_model = fit(data, log=log)
except Exception as e:
    if log:
        neptune.stop()
    print(e)
except KeyboardInterrupt:
    if log:
        neptune.stop()
