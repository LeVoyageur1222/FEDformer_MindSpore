import os
import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.train.serialization import save_checkpoint, load_checkpoint
import mindspore.ops as ops

from data_provider.data_factory_ms import data_provider
from exp.exp_basic_ms import Exp_Basic
from models.FEDformer import FEDformer
from models.Autoformer import Autoformer
from models.Informer import Informer
from models.Transformer import Transformer
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

from tqdm import tqdm


class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y):
        outputs = self.backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        loss = self.loss_fn(outputs, batch_y[:, -outputs.shape[1]:, :])
        return loss


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Informer': Informer,
            'Transformer': Transformer,
        }
        model = model_dict[self.args.model](self.args)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return nn.Adam(self.model.trainable_params(), learning_rate=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, dataset, criterion):
        total_loss = []
        self.model.set_train(False)

        for data in dataset.create_dict_iterator():
            batch_x = data['seq_x'].astype(ms.float32)
            batch_y = data['seq_y'].astype(ms.float32)
            batch_x_mark = data['seq_x_mark'].astype(ms.float32)
            batch_y_mark = data['seq_y_mark'].astype(ms.float32)
            dec_inp = ops.concat(
                (batch_y[:, :self.args.label_len, :].astype(ms.float32),
                 ops.zeros_like(batch_y[:, -self.args.pred_len:, :]).astype(ms.float32)), axis=1)
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            loss = criterion(outputs, batch_y[:, -self.args.pred_len:, :])
            total_loss.append(loss.asnumpy())

        self.model.set_train(True)
        return np.mean(total_loss)

    def train(self, setting):
        train_dataset = self._get_data('train')
        val_dataset = self._get_data('val')
        test_dataset = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        model_with_criterion = CustomWithLossCell(self.model, criterion)
        train_network = nn.TrainOneStepCell(model_with_criterion, optimizer)
        train_network.set_train()
        
        steps_per_epoch = train_dataset.get_dataset_size()  # 获取总 step 数

        for epoch in range(self.args.train_epochs):
            total_loss = []
            for data in tqdm(train_dataset.create_dict_iterator(), total=steps_per_epoch):
                batch_x = data['seq_x'].astype(ms.float32)
                batch_y = data['seq_y'].astype(ms.float32)
                batch_x_mark = data['seq_x_mark'].astype(ms.float32)
                batch_y_mark = data['seq_y_mark'].astype(ms.float32)
                dec_inp = ops.concat(
                    (batch_y[:, :self.args.label_len, :].astype(ms.float32),
                     ops.zeros_like(batch_y[:, -self.args.pred_len:, :]).astype(ms.float32)), axis=1)
                loss = train_network(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                total_loss.append(loss.asnumpy())

            train_loss = np.mean(total_loss)
            val_loss = self.vali(val_dataset, criterion)
            test_loss = self.vali(test_dataset, criterion)

            print(f"Epoch {epoch + 1}: Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f} | Test Loss {test_loss:.6f}")
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        load_checkpoint(os.path.join(path, 'checkpoint.ckpt'), self.model)
        return self.model

    def test(self, setting, test=0):
        test_dataset = self._get_data('test')
        preds, trues = [], []

        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.set_train(False)
        for data in test_dataset.create_dict_iterator():
            batch_x = data['seq_x'].astype(ms.float32)
            batch_y = data['seq_y'].astype(ms.float32)
            batch_x_mark = data['seq_x_mark'].astype(ms.float32)
            batch_y_mark = data['seq_y_mark'].astype(ms.float32)
            dec_inp = ops.concat(
                (batch_y[:, :self.args.label_len, :].astype(ms.float32),
                 ops.zeros_like(batch_y[:, -self.args.pred_len:, :]).astype(ms.float32)), axis=1)
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            preds.append(outputs.asnumpy())
            trues.append(batch_y[:, -self.args.pred_len:, :].asnumpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"Test result - MSE: {float(mse)}, MAE: {float(mae)}, RMSE: {float(rmse)}, MAPE: {float(mape)}, MSPE: {float(mspe)}")
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

    def predict(self, setting, load=False):
        pred_dataset = self._get_data('pred')
        if load:
            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.ckpt')
            load_checkpoint(best_model_path, self.model)

        preds = []
        self.model.set_train(False)
        for data in pred_dataset.create_dict_iterator():
            batch_x = data['seq_x'].astype(ms.float32)
            batch_y = data['seq_y'].astype(ms.float32)
            batch_x_mark = data['seq_x_mark'].astype(ms.float32)
            batch_y_mark = data['seq_y_mark'].astype(ms.float32)
            dec_inp = ops.concat(
                (batch_y[:, :self.args.label_len, :].astype(ms.float32),
                 ops.zeros_like(batch_y[:, -self.args.pred_len:, :]).astype(ms.float32)), axis=1)
            output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            preds.append(output.asnumpy())

        preds = np.concatenate(preds, axis=0)
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
        print(f"[Predict] Saved to {folder_path}/real_prediction.npy")
