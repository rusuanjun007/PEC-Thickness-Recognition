import haiku as hk
import numpy as np


class ConvBatchActivation(hk.Module):
    def __init__(
        self,
        output_channels,
        kernel_shape,
        stride,
        bn_decay_rate,
        activation_fn,
        bn_flag=True,
        dropoutRate=None,
    ):
        super().__init__()
        self.conv = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=stride,
            padding="SAME",
        )
        self.bn_flag = bn_flag
        if self.bn_flag:
            self.bn = hk.BatchNorm(
                create_scale=True, create_offset=True, decay_rate=bn_decay_rate
            )
        self.activation_fn = activation_fn
        self.dropoutRate = dropoutRate

    def __call__(self, x, is_training):
        x = self.conv(x)
        if self.bn_flag:
            x = self.bn(x, is_training)
        x = self.activation_fn(x)
        if (self.dropoutRate is not None) and is_training:
            x = hk.dropout(hk.maybe_next_rng_key(), self.dropoutRate, x)
        return x


class SimpleCNN2D(hk.Module):
    # Denfine model.
    def __init__(
        self,
        output_size,
        output_channels_list,
        kernel_shape,
        stride,
        bn_decay_rate,
        activation_fn,
        bn_flag=True,
        dropoutRate=None,
    ):
        super().__init__()
        if type(stride) is int:
            stride_list = np.array([[stride, 1]] * len(output_channels_list)).tolist()
        elif type(stride) is list:
            assert len(stride) == len(output_channels_list)
            stride_list = stride
        self.cba_blocks = [
            ConvBatchActivation(
                output_channels,
                kernel_shape,
                ss,
                bn_decay_rate,
                activation_fn,
                bn_flag,
                dropoutRate,
            )
            for output_channels, ss in zip(output_channels_list, stride_list)
        ]
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(output_size=output_size, with_bias=True)

    def __call__(self, x, is_training):
        for cba_block in self.cba_blocks:
            x = cba_block(x, is_training)
        x = hk.MaxPool(
            window_shape=(1, x.shape[1], 1, 1), strides=(1, 1, 1, 1), padding="VALID"
        )(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
