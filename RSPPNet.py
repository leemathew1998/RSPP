import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False, padding=1),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_BN_ReLU, self).__init__()
        batchNorm_momentum = 0.1
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum= batchNorm_momentum),
            nn.ReLU(inplace=True)
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class Resnet_Block_DILATION(nn.Module):
    def __init__(self, in_channels, out_channels,dilation):
        super(Resnet_Block_DILATION, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, bias=False,dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, bias=False,dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class UPsampling_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UPsampling_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False,dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False,dilation=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x,y):
        x = torch.cat([x,y],1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class RSPPNet(nn.Module):
    def __init__(self, num_classes=21,input_nbr=3, dilation=False,SPP=False):
        super(RSPPNet, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if dilation:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        self.block_1 = Conv_BN_ReLU(3,16)
        self.block_2 = Resnet_Block_DILATION(16,16,4)
        self.block_3 = Resnet_Block_DILATION(16,16,4)
        self.block_4 = Resnet_Block_DILATION(16,16,4)
        self.block_5 = Resnet_Block_DILATION(16,16,4)

        self.block_6 = Resnet_Block_DILATION(16,16,2)
        self.block_7 = Resnet_Block_DILATION(16,16,2)
        self.block_8 = Resnet_Block_DILATION(16,16,2)
        self.block_9 = Resnet_Block_DILATION(16,16,2)
        self.block_10 = Resnet_Block_DILATION(16,16,2)

        self.block_11 = Resnet_Block_DILATION(16,16,1)
        self.block_12 = Resnet_Block_DILATION(16,16,1)
        self.block_13 = Resnet_Block_DILATION(16,16,1)
        self.block_14 = Resnet_Block_DILATION(16,16,1)
        self.block_15 = Resnet_Block_DILATION(16,16,1)

        self.block_16 = Resnet_Block_DILATION(16,16,1)
        self.block_17 = Resnet_Block_DILATION(16,16,1)
        self.block_18 = Resnet_Block_DILATION(16,16,1)
        self.block_19 = Resnet_Block_DILATION(16,16,1)
        self.block_20 = Resnet_Block_DILATION(16,16,1)

        self.down_sample_2X = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False,dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.down_sample_4X = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False,dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4)
        )
        self.down_sample_8X = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False,dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(8),
        )

        self.compress_2 = Conv_BN_ReLU(256,16)
        self.compress_3 = Conv_BN_ReLU(512,16)
        self.compress_4 = Conv_BN_ReLU(1024,16)
        self.compress_5 = Conv_BN_ReLU(2048,16)

        self.UPsampling_Block1 = UPsampling_Block(32,32)
        self.UPsampling_Block2 = UPsampling_Block(64,64)
        self.UPsampling_Block3 = UPsampling_Block(96,96)
        self.UPsampling_Block4 = UPsampling_Block(192,192)

        self.aux = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 21, kernel_size=1),
        )
        self.morgin = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 21, kernel_size=1),
        )
        self.final = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=1),
        )
        if SPP:
            self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        else:
            self.ppm = False
        initialize_weights(self.final,self.morgin,self.aux,
        self.UPsampling_Block1,self.UPsampling_Block2,self.UPsampling_Block3,self.UPsampling_Block4,
        self.compress_2,self.compress_3,self.compress_4,self.compress_5,
        self.block_1,self.block_2,self.block_3,self.block_4,self.block_5,
        self.block_6,self.block_7,self.block_8,self.block_9,self.block_10,
        self.block_11,self.block_12,self.block_13,self.block_14,self.block_15,
        self.block_16,self.block_17,self.block_18,self.block_19,self.block_20,
        self.down_sample_2X,self.down_sample_4X,self.down_sample_8X,
        )

    def forward(self, x):
        x_size = x.size()
        # if x = 256
        fm0 = self.layer0(x)  # 128
        fm1 = self.layer1(fm0)  # 64
        compress_2 = self.compress_2(fm1)

        fm2 = self.layer2(fm1)  # 32
        compress_3 = self.compress_3(fm2)

        fm3 = self.layer3(fm2)  # 16
        compress_4 = self.compress_4(fm3)

        fm4 = self.layer4(fm3)  # 8
        compress_5 = self.compress_5(fm4)

        block_1 = self.block_1(x)
        block_2 = self.block_2(block_1)
        block_3 = self.block_3(block_2)
        block_4 = self.block_4(block_3)
        block_5 = self.block_5(block_4)

        block_6 = self.block_6(self.down_sample_2X(x))
        block_7 = self.block_7(block_6)
        block_8 = self.block_8(block_7)
        block_9 = self.block_9(block_8)
        block_10 = self.block_10(block_9)

        block_11 = self.block_11(self.down_sample_4X(x))
        block_12 = self.block_12(block_11)
        block_13 = self.block_13(block_12)
        block_14 = self.block_14(block_13)
        block_15 = self.block_15(block_14)

        block_16 = self.block_16(self.down_sample_8X(x))
        block_17 = self.block_17(block_16)
        block_18 = self.block_18(block_17)
        block_19 = self.block_19(block_18)
        block_20 = self.block_20(block_19)

        # start poit
        UPsampling_Block1 = self.UPsampling_Block1(F.upsample(compress_5, block_20.size()[2:], mode='bilinear'),block_20)
        
        UPsampling_Block2 = self.UPsampling_Block2(torch.cat([F.upsample(UPsampling_Block1, block_15.size()[2:], mode='bilinear'),F.upsample(compress_4, block_15.size()[2:], mode='bilinear')],1),block_15)
        
        UPsampling_Block3 = self.UPsampling_Block3(torch.cat([F.upsample(UPsampling_Block2, block_10.size()[2:], mode='bilinear'),F.upsample(compress_3, block_10.size()[2:], mode='bilinear')],1),block_10)

        UPsampling_Block4 = self.UPsampling_Block4(torch.cat([F.upsample(fm0, block_5.size()[2:], mode='bilinear'),
            F.upsample(UPsampling_Block3, block_5.size()[2:], mode='bilinear'),
            F.upsample(compress_2, block_5.size()[2:], mode='bilinear')],1),block_5)

        aux = self.aux(UPsampling_Block4)
        if self.ppm:
            ppm = self.ppm(fm4)
            final = self.final(ppm)
        else:
            final = self.final(fm4)
        concat = torch.cat([F.upsample(final, x_size[2:], mode='bilinear'),UPsampling_Block4],1)
        morgin = self.morgin(concat)
        if self.training:
            return morgin, aux
        else:
            return morgin
