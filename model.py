import torch
import torch.nn as nn

from torchsummary import summary

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
torch.cuda.set_device(device)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, drop=0.0, padding=1, kernel=(3, 3, 3)):
        super().__init__()

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=kernel, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_c)

        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=kernel, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_c)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout3d(p=drop)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.drop(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, drop=None):
        super().__init__()

        self.conv = ConvBlock(in_c, out_c, drop=drop)
        self.pool = nn.MaxPool3d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, drop=None):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=(2, 2, 2),
                                     stride=(2, 2, 2), padding=(0, 0, 0))
        self.conv = ConvBlock(out_c + out_c, out_c, drop=drop)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    DEFAULT_FILTERS = 64
    DEFAULT_DROPOUT = 0.05

    def __init__(self, n_channels, n_classes, n_filters=DEFAULT_FILTERS, drop=DEFAULT_DROPOUT):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.drop = drop

        """ Encoder """
        self.e1 = EncoderBlock(in_c=n_channels, out_c=n_filters, drop=drop)
        self.e2 = EncoderBlock(in_c=n_filters, out_c=n_filters * 2, drop=drop)
        self.e3 = EncoderBlock(in_c=n_filters * 2, out_c=n_filters * 4, drop=drop)
        self.e4 = EncoderBlock(in_c=n_filters * 4, out_c=n_filters * 8, drop=drop)

        """ Bottleneck """
        self.b = ConvBlock(in_c=n_filters * 8, out_c=n_filters * 16, drop=drop)

        """ Decoder """
        self.d1 = DecoderBlock(in_c=n_filters * 16, out_c=n_filters * 8, drop=drop)
        self.d2 = DecoderBlock(in_c=n_filters * 8, out_c=n_filters * 4, drop=drop)
        self.d3 = DecoderBlock(in_c=n_filters * 4, out_c=n_filters * 2, drop=drop)
        self.d4 = DecoderBlock(in_c=n_filters * 2, out_c=n_filters, drop=drop)

        """ Classifier """
        self.outputs = OutConv(in_channel=n_filters,  out_channel=n_classes)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs


if __name__ == '__main__':
    print("model")
    model = Unet(n_channels=1, n_classes=3, n_filters=64, drop=0.0)
    # print(model)
    model.to(device)
    summary(model, (1, 64, 64, 64))

