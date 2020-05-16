import tensorflow as tf
from tensorflow.keras import callbacks
import os
from datetime import datetime


class BaseTrainer:
    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data

        self.callbacks = []
        self.log_dir = None

        if self.config.trainer.tensorboard_enabled:
            experiment_name = config.trainer.get("experiment_name",
                                                 datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            directory = config.trainer.get("log_directory", "experiments/logs")
            self.log_dir = os.path.join(directory, experiment_name)

            self.file_writer = tf.summary.create_file_writer(self.log_dir)
            self.file_writer.set_as_default()
            tf.summary.text("config", self.config._text)
            if self.config.model.save_checkpoint:
                tf.summary.text("save_checkpoint", self.config.model.save_checkpoint)
            if self.config.model.load_checkpoint:
                tf.summary.text("load_checkpoint", self.config.model.load_checkpoint)

            self._init_tensorboard_callback()

        if self.config.trainer.model_checkpoint:
            self._init_model_checkpoint_callback()

    def init_callbacks(self):
        raise NotImplementedError

    def _init_tensorboard_callback(self):
        tensorboard_callback = callbacks.TensorBoard(log_dir=self.log_dir)
        self.callbacks.append(tensorboard_callback)

    def _init_model_checkpoint_callback(self):
        filepath = None
        if "filepath" in self.config.trainer.model_checkpoint_args:
            filepath = self.config.trainer.model_checkpoint_args.filepath
        elif "save_checkpoint" in self.config.model:
            filepath = self.config.model.save_checkpoint + "_training"

        self.callbacks.append(callbacks.ModelCheckpoint(
            save_weights_only=True,
            filepath=filepath
            **self.config.trainer.model_checkpoint_args
        ))

    def train(self):
        self.model.fit(
            self.data["train"],
            validation_data=self.data["valid"],
            callbacks=self.callbacks,
            epochs=self.config.trainer.epochs,
            workers=self.config.trainer.get("workers", 1),
        )
