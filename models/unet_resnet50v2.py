from base.model import BaseModel
from tensorflow.keras import layers, models, losses, metrics, optimizers, applications


def _upsample_block(channels, n=2, batch_norm=False, name=None, **kwargs):
    block = models.Sequential(name=name)
    for _ in range(n):
        block.add(layers.Conv2DTranspose(channels, 3, activation="relu", **kwargs))
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


class UnetResNet50V2Model(BaseModel):
    def build(self):
        """
        Build the model
        input shape: [batch_size, w, h, channels]
        """
        inputs = layers.Input(self.config.model.input_shape)
        x = inputs
        batch_norm = self.config.model.batch_norm

        base_model = applications.ResNet50V2(input_shape=self.config.model.input_shape, include_top=False)

        # Use the activations of these layers
        layer_names = [
            'conv1_conv',
            'conv2_block2_out', # 48 x 48 x 256
            'conv3_block3_out', # 24 x 24 x 512
            'conv4_block5_out', # 12 x 12 x 1024
            'conv5_block3_out', # 6 x 6 x 2048
        ]
        outputs = [base_model.get_layer(name).output for name in layer_names]
        output_channels = [base_model.get_layer(name).output_shape[-1] for name in layer_names]

        # Create the feature extraction model
        down_stack = models.Model(inputs=base_model.input, outputs=outputs, name="ResNet50V2")
        down_stack.trainable = False
        skips = down_stack(x)
        x = skips[-1]

        for i, (skip, k) in enumerate(zip(reversed(skips[:-1]), reversed(output_channels[:-1]))):
            upsample = _upconvolution(k, batch_norm=batch_norm, name=f"up_conv2d_{i}")
            x = upsample(x)
            concat = layers.Concatenate(axis=3)
            x = concat([skip, x])
            block = _upsample_block(k, batch_norm=batch_norm, name=f"upsampler_block_{i}", padding="same")
            x = block(x)

        output_upsample = _upconvolution(output_channels[0], batch_norm=batch_norm, name=f"up_conv2d_out")
        x = output_upsample(x)
        output_conv = layers.Conv2D(self.config.model.num_classes, 1)
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
            weighted_metrics=[metrics.CategoricalAccuracy("weighted_accuracy")],
            sample_weight_mode="temporal",
        )
