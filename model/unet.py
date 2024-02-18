__author__ = 'Nathaniel Xu'
import os
import torch
import torch.nn as nn
import torch.nn.functional as F



class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
                                        nn.ReLU(),
                                        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation),
                                        nn.ReLU())
    def forward(self, x):
        return self.conv_block(x)


class DownConv(BaseConv):
    def __init__(self, conv_args, pool_args):
        """
        Initializes a UNet down-convolution block with the following architecture: maxpool -> conv2d -> conv2d
        conv_args: dictionary of BaseConv arguments
        pool_args: dictionary of MaxPool2d arguments
        """
        super().__init__(**conv_args)
        self.maxpool = nn.MaxPool2d(**pool_args)
    def forward(self, x):
        return self.conv_block(self.maxpool(x))


class UpConv(BaseConv):
    def __init__(self, deconv_args, conv_args):
        """
        Initializes a UNet up-convolution block with the following architecture: convTranspose2d -> conv2d -> conv2d
        deconv_args: dictionary of ConvTranspose2d arguments
        conv_args:   dictionary of BaseConv arguments
        """
        super().__init__(**conv_args)
        self.upconv = nn.ConvTranspose2d(**deconv_args)
    def forward(self, x, skip):
        x = self.upconv(x)
        x_cat = torch.concat((x, skip), dim=1)
        return self.conv_block(x_cat)




class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size=(3,3), padding=(1,1))

        self.down1 = DownConv(
            conv_args={'in_channels': out_channels, 'out_channels': 2*out_channels, 'kernel_size': (3,3), 'padding': (1,1)},
            pool_args={'kernel_size': (2,2)})
        
        self.down2 = DownConv(
            conv_args={'in_channels': 2*out_channels, 'out_channels': 4*out_channels, 'kernel_size': (3,3), 'padding': (1,1)},
            pool_args={'kernel_size': (2,2)})
        
        self.down3 = DownConv(
            conv_args={'in_channels': 4*out_channels, 'out_channels': 8*out_channels, 'kernel_size': (3,3), 'padding': (1,1)},
            pool_args={'kernel_size': (2,2)})

        self.down4 = DownConv(
            conv_args={'in_channels': 8*out_channels, 'out_channels': 16*out_channels, 'kernel_size': (3,3), 'padding': (1,1)},
            pool_args={'kernel_size': (2,2)})


        self.up4 = UpConv(
            deconv_args={'in_channels': 16*out_channels, 'out_channels': 8*out_channels, 'kernel_size': (2,2), 'padding': (0,0), 'stride': (2,2)},
            conv_args={'in_channels': 2*8*out_channels, 'out_channels': 8*out_channels, 'kernel_size': (3,3), 'padding': (1,1)})

        self.up3 = UpConv(
            deconv_args={'in_channels': 8*out_channels, 'out_channels': 4*out_channels, 'kernel_size': (2,2), 'padding': (0,0), 'stride': (2,2)},
            conv_args={'in_channels': 2*4*out_channels, 'out_channels': 4*out_channels, 'kernel_size': (3,3), 'padding': (1,1)})

        self.up2 = UpConv(
            deconv_args={'in_channels': 4*out_channels, 'out_channels': 2*out_channels, 'kernel_size': (2,2), 'padding': (0,0), 'stride': (2,2)},
            conv_args={'in_channels': 2*2*out_channels, 'out_channels': 2*out_channels, 'kernel_size': (3,3), 'padding': (1,1)})

        self.up1 = UpConv(
            deconv_args={'in_channels': 2*out_channels, 'out_channels': out_channels, 'kernel_size': (2,2), 'padding': (0,0), 'stride': (2,2)},
            conv_args={'in_channels': 2*out_channels, 'out_channels': out_channels, 'kernel_size': (3,3), 'padding': (1,1)})

        self.head_logits = nn.Conv2d(out_channels, num_classes, kernel_size=(1,1))      # nn.Conv2d(out_channels, num_classes, kernel_size=(3,3), padding=(1,1))
        self.head_probs = nn.LogSoftmax(dim=1) if num_classes>1 else nn.LogSigmoid()


    def forward(self, x):
        # Encoder
        d0 = self.init_conv(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        # Decoder
        u3 = self.up4(d4, d3)
        u2 = self.up3(u3, d2)
        u1 = self.up2(u2, d1)
        u0 = self.up1(u1, d0)
        logits = self.head_logits(u0)
        return self.head_probs(logits), logits


    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        #self.eval()






