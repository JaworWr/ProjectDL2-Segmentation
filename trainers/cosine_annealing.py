from base.trainer import BaseTrainer
import tensorflow as tf
from keras import callbacks
import numpy as np
from utils import data

class CosineAnnealingCallback(callbacks.Callback):
    def __init__(self, lr_min, lr_max, run_lengths, n_batches):
        super().__init__()
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.run_lengths = run_lengths
        self.n_batches = n_batches

        self.run_epoch = 0
        self.run_idx = 0
        self.run_len = run_lengths[0]

    def on_epoch_begin(self, epoch, logs=None):
        self.run_epoch += 1
        if self.run_len < self.run_epoch:
            self.run_idx += 1
            self.run_epoch = 0
            self.run_len = self.run_lengths[self.run_idx]

    def on_train_batch_begin(self, batch, logs=None):
        t = self.run_epoch + batch / self.n_batches
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * t / self.run_len))
        self.model.optimizer.learning_rate = lr


class CosineAnnealingTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.init_callbacks()

    def log_lr(self, epoch):
        tf.summary.scalar("learning_rate", data=self.model.model.optimizer.lr, step=epoch)

    def init_callbacks(self):
        self.callbacks.append(CosineAnnealingCallback(
            lr_min=self.config.trainer.lr_min,
            lr_max=self.config.trainer.lr_max,
            run_lengths=self.config.trainer.run_lengths,
            n_batches=data.get_train_batch_count(),
        ))

        if self.config.trainer.tensorboard_enabled:
            self.callbacks.append(callbacks.LambdaCallback(
                on_epoch_begin=lambda epoch, loss: self.log_lr(epoch)
            ))
