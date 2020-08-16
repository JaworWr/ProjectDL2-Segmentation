from base.model import BaseModel
from utils.data import N_CLASSES
from tensorflow.keras import layers, models, losses, metrics, optimizers


def _downsample_block(channels, n=2, batch_norm=False, name=None, **kwargs):
    block = models.Sequential(name=name)
    for _ in range(n):
        block.add(layers.Conv2D(channels, 3, activation="relu", **kwargs))
        if batch_norm:
            block.add(layers.BatchNormalization())
    return block


def _upsample_block(channels, n=2, batch_norm=False, name=None):
    block = models.Sequential(name=name)
    for _ in range(n):
        block.add(layers.Conv2DTranspose(channels, 3, padding="valid", activation="relu"))
        if batch_norm:
            block.add(layers.BatchNormalization())
    return block


def _upconvolution(channels, batch_norm=False, name=None):
    block = models.Sequential(name=name)
    block.add(layers.UpSampling2D(2))
    block.add(layers.Conv2D(channels, 2, padding="same"))
    if batch_norm:
        block.add(layers.BatchNormalization())
    return block


class UnetModel(BaseModel):
    def build(self):
        """
        Build the model
        input shape: [batch_size, w, h, channels]
        """
        inputs = layers.Input(self.config.model.input_shape)
        x = inputs
        skips = []
        channels = self.config.model.channels
        batch_norm = self.config.model.batch_norm

        for i, k in enumerate(channels[:-1]):
            block = _downsample_block(k, batch_norm=batch_norm, padding="valid", name=f"downsampler_block_{i}")
            x = block(x)
            skips.append(x)
            pool = layers.MaxPool2D(2)
            x = pool(x)

        bottom_block = _downsample_block(channels[-1], batch_norm=batch_norm, padding="same", name="bottom_block")
        x = bottom_block(x)

        for i, (skip, k) in enumerate(zip(reversed(skips), reversed(channels[:-1]))):
            upsample = _upconvolution(k, batch_norm=batch_norm, name=f"up_conv2d_{i}")
            x = upsample(x)
            concat = layers.Concatenate(axis=3)
            x = concat([skip, x])
            block = _upsample_block(k, batch_norm=batch_norm, name=f"upsampler_block_{i}")
            x = block(x)

        output_conv = layers.Conv2D(self.config.model.num_classes, 1, activation="softmax")
        x = output_conv(x)

        flatten = layers.Reshape((-1, self.config.model.num_classes))
        outputs = flatten(x)
        self.model = models.Model(inputs=inputs, outputs=outputs)

    def compile(self):
        if not self.model:
            raise RuntimeError("You have to build the model before compiling it.")

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.model.learning_rate),
            loss=losses.CategoricalCrossentropy(from_logits=True),
            metrics=[metrics.CategoricalAccuracy(name="accuracy")],
            weighted_metrics=[metrics.MeanIoU(N_CLASSES, name="weighted_iou")],
            sample_weight_mode="temporal",
        )
