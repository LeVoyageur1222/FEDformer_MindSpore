import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
# import mindspore.save_checkpoint as save_checkpoint
from mindspore.train.serialization import save_checkpoint
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')  # 不弹窗绘图，直接保存

def adjust_learning_rate(optimizer, epoch, args):
    """
    Dynamic learning rate adjustment
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        new_lr = lr_adjust[epoch]
        # 更新MindSpore Optimizer学习率
        if hasattr(optimizer, 'learning_rate'):
            if isinstance(optimizer.learning_rate, ms.Tensor):
                optimizer.learning_rate.set_data(ms.Tensor(new_lr, dtype=ms.float32))
            else:
                optimizer.learning_rate = new_lr
        print(f"Updating learning rate to {new_lr}")

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        save_checkpoint(model, f"{path}/checkpoint.ckpt")
        self.val_loss_min = val_loss

class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler:
    """
    Standard scaler: (x - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-5)

    def inverse_transform(self, data):
        return (data * (self.std + 1e-5)) + self.mean

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()
