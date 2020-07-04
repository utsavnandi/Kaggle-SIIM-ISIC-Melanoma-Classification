import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_roc(y_true, y_pred, show=False):
    testy, lr_probs = y_true, y_pred
    ns_probs = [0 for _ in range(len(testy))]
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)  # lr_probs: predictions
    # plot the roc curve for the model
    figure = plt.figure(figsize=(8, 8))
    plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
    plt.plot(lr_fpr, lr_tpr, linestyle="-", label="Model")
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # show the legend
    plt.legend()
    # show the plot
    if show:
        plt.show()
    else:
        plt.clf()
    return figure


def draw_hist(y_true, y_pred, show=True):
    true_neg_indices = np.where(y_true[:, 0] == 0)[0]
    true_pos_indices = np.where(y_true[:, 0] == 1)[0]
    pred_true_pos = y_pred[true_pos_indices]
    pred_true_neg = y_pred[true_neg_indices]
    thresh = 0.2
    pred_true_pos_error_count = pred_true_pos[
        np.where(pred_true_pos < (1 - thresh))[0]
    ]
    pred_true_neg_error_count = pred_true_neg[
        np.where(pred_true_neg > (thresh))[0]
    ]
    total_error_count = (
        pred_true_pos_error_count.shape[0] + pred_true_neg_error_count.shape[0]
    )
    total_error_count_scaled = total_error_count / y_true.shape[0]
    # plot
    figure = plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(121)
    ax1.hist(pred_true_neg.T[0], bins=10)
    plt.ylim((0, 2000))
    ax2 = plt.subplot(122)
    ax2.hist(pred_true_pos.T[0], bins=10)
    if show:
        plt.show()
    else:
        plt.clf()
    return figure, total_error_count_scaled
