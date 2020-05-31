from base.trainer import BaseTrainer
import tensorflow as tf
from keras import callbacks
import numpy as np
from utils.data import get_train_batch_count

class CosineAnnealingCallback(callbacks.Callback):
    def __init__(self, lr_min, lr_max, run_mult, n_batches):
        super().__init__()
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.run_mult = run_mult
        self.n_batches = n_batches

        self.run_epoch = 0
        self.run_len = 1

    def on_epoch_end(self, epoch, logs=None):
        self.run_epoch += 1
        if self.run_epoch >= self.run_len:
            self.run_epoch = 0
            self.run_len *= self.run_mult

    def on_train_batch_begin(self, batch, logs=None):
        t = self.run_epoch + batch / self.n_batches
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * t / self.run_len))
        self.model.optimizer.lr = lr


class CosineAnnealingTrainer(BaseTrainer):
    def __init__(self, config, model, data):
        super().__init__(config, model, data)
        self.n_batches = get_train_batch_count(self.config)
        self.epoch = 0
        self.init_callbacks()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def log_lr(self, batch):
        batch += self.epoch * self.n_batches
        tf.summary.scalar("learning_rate", data=self.model.model.optimizer.lr, step=batch)

    def init_callbacks(self):
        self.callbacks.append(CosineAnnealingCallback(
            lr_min=self.config.trainer.lr_min,
            lr_max=self.config.trainer.lr_max,
            run_mult=self.config.trainer.run_mult,
            n_batches=self.n_batches,
        ))

        if self.config.trainer.tensorboard_enabled:
            self.callbacks.append(callbacks.LambdaCallback(
                on_epoch_begin=lambda epoch, logs: self.set_epoch(epoch),
                on_batch_begin=lambda batch, logs: self.log_lr(batch),
            ))
