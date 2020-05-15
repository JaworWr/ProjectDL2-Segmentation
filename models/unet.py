from base.model import BaseModel
from tensorflow.keras import layers, models, losses, metrics, optimizers


def _downsample_block(channels, n=2, batch_norm=False, name=None):
    block = models.Sequential(name=name)
    for _ in range(n):
        block.add(layers.Conv2D(channels, 3, padding="valid", activation="relu"))
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
    block =  models.Sequential(name=name)
    block.add(layers.UpSampling2D(2))
    block.add(layers.Conv2D(channels, 2, padding="same"))
    if batch_norm:
        block.add(layers.BatchNormalization())
    return block

class UnetModel(BaseModel):
    def build(self):
        """
        input shape: [batch_size, w, h, channels]
        """
        inputs = input(self.config.model.input_shape)
        x = inputs
        skips = []
        channels = self.config.model.channels
        batch_norm = self.config.model.batch_norm

        for k in channels[:-1]:
            block = _downsample_block(k, batch_norm=batch_norm, name="downsampler_block")
            x = block(x)
            skips.append(x)
            pool = layers.MaxPool2D(2)
            x = pool(x)

        bottom_conv = layers.Conv2D(channels[-1], 3, padding="same", activation="relu")
        x = bottom_conv(x)
        if batch_norm:
            batch_norm = layers.BatchNormalization()
            x = batch_norm(x)

        for skip, k in zip(reversed(skips), reversed(channels[:-1])):
            upsample = _upconvolution(k, batch_norm=batch_norm, name="up-conv2d")
            x = upsample(x)
            concat = layers.Concatenate(axis=3)
            x = concat([skip, x])
            block = _upsample_block(k, batch_norm=batch_norm, name="upsampler_block")
            x = block(x)

        outputs = layers.Conv2D(self.config.model.num_classes, 1)
        self.model = models.Model(inputs=inputs, outputs=outputs)

    def compile(self):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.model.learning_rate),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[metrics.SparseCategoricalAccuracy()],
        )