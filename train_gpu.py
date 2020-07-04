# import cv2
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader  # WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from omegaconf import OmegaConf

# from ranger import Ranger
# import neptune
from augmentations import get_train_transforms, get_valid_transforms
from dataset import MelanomaDataset
from loss import sigmoid_focal_loss

# from metric import RocAucMeter
from models import EfficientNet
from plots import plot_roc, draw_hist


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    optimizer.zero_grad()
    for idx, (images, targets) in tqdm(enumerate(loader), total=len(loader)):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        y_pred = model(images.float())
        loss = (
            sigmoid_focal_loss(y_pred, targets, FLAGS["alpha"], FLAGS["gamma"])
            / FLAGS["accumulation_steps"]
        )
        running_loss += float(loss)

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (idx + 1) % FLAGS["accumulation_steps"] == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        # if log and (idx + 1) % FLAGS["log_interval"] == 0:
        #    neptune.log_metric("Loss/train", float(loss))

    return running_loss / len(loader)


def val_one_epoch(loader, model):
    model.eval()
    running_loss = 0.0
    y_preds_list = []
    targets_list = []
    with torch.no_grad():
        for idx, (images, targets) in tqdm(
            enumerate(loader), total=len(loader)
        ):
            images = images.to(device)
            targets = targets.to(device)  # .unsqueeze(1)
            y_pred = model(images.float())
            loss = sigmoid_focal_loss(
                y_pred, targets, FLAGS["alpha"], FLAGS["gamma"]
            )
            running_loss += float(loss)
            y_preds_list.append(torch.sigmoid(y_pred).cpu().numpy())
            targets_list.append(targets.cpu().numpy())
        y_true = np.vstack(targets_list)
        y_pred = np.vstack(y_preds_list)
        auc_score = roc_auc_score(
            y_true, y_pred
        )  # add [:, 1] for cross entropy
        roc_plot = plot_roc(y_true, y_pred)  # add [:, 1] for cross entropy
        hist, error_scaled = draw_hist(y_true, y_pred)
        print(f"roc_auc_score: {auc_score:.5f}")
        print(f"average loss for val epoch: {running_loss/len(loader):.5f}")
        print(f"scaled error: {error_scaled:.5f}")
    return running_loss / len(loader), auc_score, roc_plot, hist, error_scaled


def save_upload(
    model, optimizer, best_score, epoch, fold=None, exp_name="model"
):
    if fold:
        NAME = (
            "siim-isic_"
            + exp_name
            + f"_fold_{str(fold+1)}_{str(epoch+1)}.ckpt"
        )
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
    print(f"Saved ckpt for epoch {epoch+1}, score: {best_score:.5f}")


def fit(data, fold=None, log=True):

    best_score = 0.0
    model = EfficientNet("tf_efficientnet_b0_ns").to(device)
    # model.load_state_dict(
    #     torch.load("/content/siim-isic_efficientnet_b0_2.ckpt")[
    #         "model_state_dict"
    #     ]
    # )
    # if log:
    #    neptune.init("utsav/SIIM-ISIC", api_token=NEPTUNE_API_TOKEN)
    #    neptune.create_experiment(
    #        FLAGS["exp_name"],
    #        exp_description,
    #        params=FLAGS,
    #        upload_source_files="*.txt",
    #    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FLAGS["learning_rate"],
        weight_decay=FLAGS["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        cooldown=0,
        mode="min",
        patience=3,
        verbose=True,
        min_lr=1e-8,
    )

    datasets = get_datasets(data)

    # sampler
    # labels_vcount = y_train["target"].value_counts()
    # class_counts = [
    #     labels_vcount[0].astype(np.float32),
    #     labels_vcount[1].astype(np.float32),
    # ]
    # num_samples = sum(class_counts)
    # class_weights = [
    #     num_samples / class_counts[i] for i in range(len(class_counts))
    # ]
    # weights = [
    #     class_weights[y_train["target"].values[i]]
    #     for i in range(int(num_samples))
    # ]
    # sampler = WeightedRandomSampler(
    #     torch.DoubleTensor(weights), int(num_samples)
    # )

    # loaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=FLAGS["batch_size"],
        num_workers=FLAGS["num_workers"],
        shuffle=True,  # sampler=sampler,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["valid"],
        batch_size=FLAGS["batch_size"] * 2,
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

        print(f"\nAverage loss for epoch #{epoch+1} : {train_loss:.5f}")
        val_output = val_one_epoch(val_loader, model)
        val_loss, auc_score, roc_plot, hist, error_scaled = val_output
        scheduler.step(error_scaled)

        # logs
        # if log:
        #     neptune.log_metric("AUC/val", auc_score)
        #     neptune.log_image("ROC/val", roc_plot)
        #     neptune.log_metric("Loss/val", val_loss)
        #     neptune.log_image("hist/val", hist)

        # checkpoint+upload
        if (auc_score > best_score) or (best_score - auc_score < 0.025):
            if auc_score > best_score:
                best_score = auc_score
            save_upload(
                model,
                optimizer,
                best_score,
                epoch,
                fold,
                exp_name=FLAGS["exp_name"],
            )

        print("-" * 28 + f"Epoch #{epoch+1} ended" + "-" * 28)

    # if log:
    #    neptune.stop()

    return model


if __name__ == "__main__":
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    seed_everything(43)
    cfg = OmegaConf.load("./gpu_train_config.yaml")
    print(cfg.pretty())
    FLAGS = cfg.FLAGS
    IMG_SIZE = FLAGS.img_size
    log = False
    DATA_DIR = cfg.data_dir
    df_train = pd.read_csv(DATA_DIR + "folds_13062020.csv")
    df_test = pd.read_csv(DATA_DIR + "test.csv").rename(
        columns={"image_name": "image_id"}
    )
    sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")

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
