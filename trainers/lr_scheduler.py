import tensorflow as tf
from tensorflow.keras import callbacks

from base.trainer import BaseTrainer


class LRSchedulerTrainer(BaseTrainer):
    def __init__(self, config, model, data):
        super().__init__(config, model, data)
        self.init_callbacks()

    def log_lr(self, epoch):
        tf.summary.scalar("learning_rate", data=self.model.model.optimizer.lr, step=epoch)

    def init_callbacks(self):
        def schedule(epoch):
            for step in self.config.trainer.lr_schedule:
                if epoch < step.until_epoch:
                    return step.lr

        self.callbacks.append(callbacks.LearningRateScheduler(schedule))
        if self.config.trainer.tensorboard_enabled:
            self.callbacks.append(callbacks.LambdaCallback(
                on_epoch_begin=lambda epoch, loss: self.log_lr(epoch)
            ))
